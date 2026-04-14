import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_config():
    base_dir = Path(__file__).resolve().parent.parent
    return {
        "BASE_DIR": str(base_dir),
        "MEMORY_FILE": str(base_dir / "memory.json"),
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
        "GROQ_MODEL": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "GROQ_COMPLEX_MODEL": os.environ.get("GROQ_COMPLEX_MODEL", "llama-3.3-70b-versatile"),
        "GROQ_MAX_TOKENS": _as_int(os.environ.get("GROQ_MAX_TOKENS"), 160),
        "GROQ_TEMPERATURE": _as_float(os.environ.get("GROQ_TEMPERATURE"), 0.3),
        "GROQ_MAX_RETRIES": _as_int(os.environ.get("GROQ_MAX_RETRIES"), 2),
        "GROQ_COOLDOWN_SECONDS": _as_int(os.environ.get("GROQ_COOLDOWN_SECONDS"), 20),
        "SYSTEM_PROMPT_STYLE": os.environ.get("SYSTEM_PROMPT_STYLE", "voice_assistant"),
        "GREETING_NAME": os.environ.get("GREETING_NAME", "Shanky"),
        "START_WITH_GREETING": _as_bool(os.environ.get("START_WITH_GREETING"), True),
        "STT_PROVIDER": os.environ.get("STT_PROVIDER", "auto"),
        "STT_MODEL": os.environ.get("STT_MODEL", "tiny"),
        "TTS_PROVIDER": os.environ.get("TTS_PROVIDER", "auto"),
        "TTS_VOICE": os.environ.get("TTS_VOICE", ""),
        "PIPER_COMMAND": os.environ.get("PIPER_COMMAND", "piper"),
        "PIPER_MODEL_PATH": os.environ.get("PIPER_MODEL_PATH", ""),
        "PIPER_CONFIG_PATH": os.environ.get("PIPER_CONFIG_PATH", ""),
        "VAD_PROVIDER": os.environ.get("VAD_PROVIDER", "auto"),
        "WAKE_WORD_PROVIDER": os.environ.get("WAKE_WORD_PROVIDER", "auto"),
        "WAKE_WORD": os.environ.get("WAKE_WORD", "hey assistant"),
        "WAKE_WORD_MODEL_PATH": os.environ.get("WAKE_WORD_MODEL_PATH", ""),
        "USE_WAKE_WORD": _as_bool(os.environ.get("USE_WAKE_WORD"), True),
        "BYPASS_WAKE_WORD": _as_bool(os.environ.get("BYPASS_WAKE_WORD"), False),
        "AUTO_CALIBRATE_NOISE": _as_bool(os.environ.get("AUTO_CALIBRATE_NOISE"), True),
        "NOISE_REDUCTION": _as_bool(os.environ.get("NOISE_REDUCTION"), True),
        "DEBUG_AUDIO": _as_bool(os.environ.get("DEBUG_AUDIO"), False),
        "DEBUG_AUDIO_INTERVAL_MS": _as_int(os.environ.get("DEBUG_AUDIO_INTERVAL_MS"), 1000),
        "AUDIO_SAMPLE_RATE": _as_int(os.environ.get("AUDIO_SAMPLE_RATE"), 16000),
        "AUDIO_CHANNELS": _as_int(os.environ.get("AUDIO_CHANNELS"), 1),
        "CHUNK_MS": _as_int(os.environ.get("CHUNK_MS"), 80),
        "MAX_UTTERANCE_SECONDS": _as_int(os.environ.get("MAX_UTTERANCE_SECONDS"), 12),
        "END_OF_UTTERANCE_SILENCE_MS": _as_int(os.environ.get("END_OF_UTTERANCE_SILENCE_MS"), 450),
        "IDLE_TIMEOUT_SECONDS": _as_int(os.environ.get("IDLE_TIMEOUT_SECONDS"), 45),
        "MAX_LONG_TERM_SUMMARIES": _as_int(os.environ.get("MAX_LONG_TERM_SUMMARIES"), 6),
        "MIC_DEVICE": os.environ.get("MIC_DEVICE"),
        "SPEAKER_DEVICE": os.environ.get("SPEAKER_DEVICE"),
        "ENABLE_BROWSER_REALTIME": _as_bool(os.environ.get("ENABLE_BROWSER_REALTIME"), False),
    }
