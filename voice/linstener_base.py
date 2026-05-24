from abc import ABC, abstractmethod
from config.settings import settings

class Listener(ABC):
    """
    Abstract base class for speech-to-text listeners.

    Subclasses implement initialise_model() and listen().
    Everything else — lifecycle, context manager — lives here.
    """
    def __init__(self,
       sample_rate: int = settings.audio.sample_rate
    ):
        self.sample_rate = sample_rate
        self.initialise_model()
    
    @abstractmethod
    def initialise_model(self):
        """Load all models required for STT. Called once during __init__."""
        pass 
    
    @abstractmethod
    def listen(self) -> str:
        """Block until a full utterance is captured. Return transcribed text."""
        pass 
    
    def close(self):
        pass 
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.close()