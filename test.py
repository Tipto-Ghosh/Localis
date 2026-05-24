import torch
import numpy as np
import sounddevice as sd
from transformers import pipeline


# print("Available Audio Devices:\n")
# print(sd.query_devices())
# print(" - " * 15)

# default_input_device_index = sd.default.device[0]
# device_info = sd.query_devices(default_input_device_index)

# print("\nUsing Input Device:")
# print(device_info)
# print(" - " * 15)

pipe = pipeline(
    "automatic-speech-recognition",
    model = "distil-whisper/distil-small.en",
    dtype = torch.float16,
    device = "cuda",
)

SAMPLE_RATE = 16000
DURATION = 10  # seconds to record

print("Recording... speak now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate = SAMPLE_RATE,
    channels = 1,
    dtype = "float32"
)
sd.wait()
print("Done recording.")

audio = audio.squeeze()  # (samples, 1) → (samples,)
result = pipe(audio)
print("Transcription:", result["text"])