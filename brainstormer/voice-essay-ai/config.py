# config.py - Configuration Settings
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Flask settings
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Voice processing settings
VOICE_CONFIG = {
    'whisper_model': 'base',  # Options: tiny, base, small, medium, large
    'tts_rate': 150,         # Words per minute
    'audio_timeout': 30,     # Max audio processing time
    'max_file_size': 10 * 1024 * 1024,  # 10MB max audio file
}

# AI settings
AI_CONFIG = {
    'ollama_url': 'http://localhost:11434',
    'default_model': 'mistral:7b',
    'fallback_model': 'tinyllama',
    'max_conversation_length': 16,
    'response_timeout': 60,
}

# Data storage
DATA_CONFIG = {
    'conversations_file': BASE_DIR / 'data' / 'conversations.json',
    'essays_file': BASE_DIR / 'data' / 'essays.json',
    'audio_temp_dir': BASE_DIR / 'data' / 'audio_temp',
    'models_cache_dir': BASE_DIR / 'models',
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs' / 'app.log',
            'maxBytes': 1024*1024*15,  # 15MB
            'backupCount': 5,
            'formatter': 'default',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console'],
    },
}

# Create directories if they don't exist
for path in [
    DATA_CONFIG['audio_temp_dir'],
    DATA_CONFIG['models_cache_dir'],
    BASE_DIR / 'logs',
    BASE_DIR / 'data',
]:
    path.mkdir(parents=True, exist_ok=True)