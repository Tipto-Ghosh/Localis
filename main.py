import warnings
warnings.filterwarnings("ignore")

from voice.listener import WhisperListener
from voice.speaker import KokoroVoice

listener = WhisperListener(wake_word_threshold=0.3)
voice    = KokoroVoice()

print("Start Talking...\n")

while True:
    text = listener.listen()
    print("You said:", text)

    if text:
        voice.speak(text)