from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd
from config.settings import settings


class Voice(ABC):
    """Abstract base class for text-to-speech.
    Owns the audio output stream and the speak() pipeline.
    Subclasses only need to implement initialise_model() and
    convert_text_to_speech() - everything else is handled here.
    """
    def __init__(self,
        sample_rate: int = settings.audio.sample_rate,
        chunk_size: int = settings.audio.chunk_size
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        self.output_stream = sd.OutputStream(
            samplerate = self.sample_rate,
            channels = 1,
            dtype = "float32"
        )
        self.output_stream.start()
        self.initialise_model()
    
    @abstractmethod
    def initialise_model(self):
        """Load the TTS model. Called once during __init__."""
        pass 
    
    @abstractmethod
    def convert_text_to_speech(self, text: str) -> list[np.ndarray]:
        """Convert text to a list of float32 audio chunks."""
        pass 
    
    def speak(self, text: str):
        """Speak the provided text through the device output."""
        frames = self.convert_text_to_speech(text)
        for frame in frames:
            self.output_stream.write(frame)
    
    def close(self):
        """Stop and release the audio output."""
        self.output_stream.stop()
        self.output_stream.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.close()