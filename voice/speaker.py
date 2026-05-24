import numpy as np
from kokoro import KPipeline
from voice.speaker_base import Voice
from config.settings import settings


class KokoroVoice(Voice):
    """
    Kokoro-backed TTS implementation.

    Uses the KPipeline from the kokoro package to synthesise speech.
    See https://github.com/hexgrad/kokoro for available voices.
    """
    def __init__(
        self,
        voice: str = settings.voice.kokoro_voice,
        sample_rate: int = settings.audio.sample_rate,
        chunk_size: int = settings.audio.chunk_size,
    ):
        self.voice = voice
        super().__init__(sample_rate, chunk_size)

    def initialise_model(self):
        """Load the Kokoro pipeline."""
        self.pipeline = KPipeline(
            lang_code = settings.voice.kokoro_lang_code,
            repo_id = settings.voice.kokoro_repo_id
        )

    def convert_text_to_speech(self, text: str) -> list[np.ndarray]:
        """
        Run the Kokoro pipeline and split the output into chunk_size frames.

        KPipeline yields (graphemes, phonemes, audio) per sentence segment.
        We flatten all segments and chunk into fixed-size numpy arrays so
        speak() can stream them incrementally to the output device.
        """
        frames = []

        for _, _, audio in self.pipeline(text, voice=self.voice):
            # audio is a 1-D torch tensor — detach before converting
            waveform = audio.detach().cpu().numpy().astype(np.float32)

            for start in range(0, len(waveform), self.chunk_size):
                frames.append(waveform[start : start + self.chunk_size])

        return frames