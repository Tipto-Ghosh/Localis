import numpy as np
import torch
import sounddevice as sd
from transformers import pipeline
from openwakeword.model import Model as WakeWordModel
from silero_vad import load_silero_vad, VADIterator
from voice.linstener_base import Listener
from config.settings import settings

class WhisperListener(Listener):
    """
    Full listening pipeline:
      OpenWakeWord -> Silero VAD -> distil-whisper/distil-small.en

    Phase 1 — _wait_for_wake_word():
        Streams 1280-sample chunks (80ms) to OpenWakeWord on CPU.
        Blocks until the wake word score exceeds the threshold.

    Phase 2 — _record_until_silence():
        Switches to 512-sample chunks required by Silero VAD.
        Accumulates frames until speech ends (silence_duration of
        consecutive silence after at least one speech frame).

    Phase 3 — listen():
        Passes the accumulated audio to the Whisper pipeline on GPU.
        Returns the stripped transcription string.
    """
    _WAKEWORD_CHUNK = 1_280  # 80ms  @ 16kHz  OpenWakeWord requirement
    _VAD_CHUNK = 512     # 32ms  @ 16kHz  Silero VAD requirement

    def __init__(
        self,
        wake_word: str = settings.stt.wake_words,
        wake_word_threshold: float = 0.5,
        silence_duration: float = 1.2,    # seconds of silence before stopping
        max_record_seconds: float = 15.0,   # hard cap to avoid infinite recording
        mic_gain: float = 3.0
    ):
        # OpenWakeWord model names use underscores: "hey jarvis" -> "hey_jarvis"
        self.wake_word = wake_word.strip().replace(" ", "_")
        self.wake_word_threshold = wake_word_threshold
        self.silence_duration = silence_duration
        self.max_record_seconds = max_record_seconds
        self.mic_gain = mic_gain
        super().__init__(sample_rate = 16_000)

    def initialise_model(self):
        """Load OpenWakeWord, Silero VAD, and the Whisper ASR pipeline."""
        print("[Listener] Loading wake word model...")
        self.wakeword_model = WakeWordModel(
            wakeword_models = [self.wake_word],
            inference_framework = "onnx",
        )

        print("[Listener] Loading Silero VAD...")
        _vad = load_silero_vad()
        self.vad_iterator = VADIterator(
            model=_vad,
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_silence_duration_ms=300,
        )

        print("[Listener] Loading Whisper ASR pipeline...")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="distil-whisper/distil-small.en",
            dtype=torch.float16,
            device="cuda",
        )
        print("[Listener] Ready.")

    def listen(self) -> str:
        """
        Full pipeline: wake word → record → transcribe.
        Returns the transcribed text string.
        """
        self._wait_for_wake_word()
        audio = self._record_until_silence()
        result = self.asr_pipeline(audio)
        return result["text"].strip()

    def _wait_for_wake_word(self):
        """Stream audio to OpenWakeWord until the wake word is detected."""
        print(f"[Listener] Waiting for wake word: '{self.wake_word}'")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._WAKEWORD_CHUNK,
        ) as stream:
            while True:
                chunk, _ = stream.read(self._WAKEWORD_CHUNK)
                chunk = chunk.squeeze()

                # OpenWakeWord expects int16 — convert from float32
                chunk_amplified = np.clip(chunk * self.mic_gain, -1.0, 1.0)
                chunk_int16 = (chunk_amplified * 32_768).astype(np.int16)
                self.wakeword_model.predict(chunk_int16)

                score = self.wakeword_model.prediction_buffer[self.wake_word][-1]
                if score > 0.1:
                    print(f"[WakeWord] score: {score:.3f}")
                if score >= self.wake_word_threshold:
                    print("[Listener] Wake word detected!")
                    return

    def _record_until_silence(self) -> np.ndarray:
        """
        Record audio until Silero VAD detects end-of-speech.

        Accumulates frames from the moment recording starts.
        Stops when silence_duration seconds of consecutive silence
        follows at least one detected speech segment, or when
        max_record_seconds is reached.
        """
        print("[Listener] Recording...")

        frames = []
        silence_frames = 0
        speech_started = False

        silence_limit = int(
            self.silence_duration * self.sample_rate / self._VAD_CHUNK
        )
        max_frames = int(
            self.max_record_seconds * self.sample_rate / self._VAD_CHUNK
        )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._VAD_CHUNK,
        ) as stream:
            for _ in range(max_frames):
                chunk, _ = stream.read(self._VAD_CHUNK)
                chunk = chunk.squeeze()
                chunk - np.clip(chunk * self.mic_gain, -1.0, 1.0)
                frames.append(chunk)

                vad_result = self.vad_iterator(
                    torch.from_numpy(chunk),
                    return_seconds=False,
                )

                if vad_result and "start" in vad_result:
                    speech_started = True
                    silence_frames = 0

                elif vad_result and "end" in vad_result and speech_started:
                    silence_frames += 1
                    if silence_frames >= silence_limit:
                        break

        self.vad_iterator.reset_states()
        print("[Listener] Recording complete.")
        return np.concatenate(frames)