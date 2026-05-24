import os 
from dataclasses import dataclass,  field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen = True)
class AudioSettings:
    sample_rate: int = 16000 # kokoro sample rate
    chunk_size: int = 2048
    
# Voice/Text-to-Speech
@dataclass(frozen = True)
class VoiceSettings:
    kokoro_voice: str = "af_heart" # American female
    kokoro_lang_code: str = "a" # American English
    kokoro_repo_id: str = "hexgrad/Kokoro-82M"

# Speech-to-Text
@dataclass(frozen = True)
class STTSettings:
    wake_words: str = os.getenv("WAKE_WORDS", "hey jarvis")
    whisper_model: str = "distil-large-v3"
    language: str = "en"
    
# LLM Agent
@dataclass(frozen = True)
class AgentSettings:
    ollama_model: str = os.getenv("OLLAMA_MODEL" , "qwen2.5:7b-instruct-q4_K_M")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@dataclass(frozen = True)
class WeatherSettings:
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    default_city: str = os.getenv("DEFAULT_CITY", "Dhaka")

@dataclass(frozen=True)
class SearchSettings:
    max_results: int = 5

# MCP server
@dataclass(frozen = True)
class MCPSettings:
    host: str = "localhost"
    port: int = 6789

@dataclass(frozen = True)
class Settings:
    audio: AudioSettings = field(default_factory = AudioSettings)
    voice: VoiceSettings = field(default_factory = VoiceSettings)
    stt: STTSettings = field(default_factory = STTSettings)
    agent: AgentSettings = field(default_factory = AgentSettings)
    weather: WeatherSettings = field(default_factory = WeatherSettings)
    search: SearchSettings = field(default_factory = SearchSettings)
    mcp: MCPSettings = field(default_factory = MCPSettings)
    
settings = Settings()