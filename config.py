# Configuration file for Advanced Audio Transcription - Whisper Powered

import os
from pathlib import Path

# =======================
# Whisper Model Configuration
# =======================

# Whisper model size - impacts accuracy vs speed tradeoff
# Options: "tiny", "base", "small", "medium", "large-v3"
# For Hindi: "large-v3" gives BEST accuracy, "medium" is good balance
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")

# Use faster-whisper for 4x speed improvement
USE_FASTER_WHISPER = os.getenv("USE_FASTER_WHISPER", "true").lower() == "true"

# Enable audio enhancement (noise reduction, normalization)
ENABLE_AUDIO_ENHANCEMENT = os.getenv("ENABLE_AUDIO_ENHANCEMENT", "true").lower() == "true"



# =======================
# Audio Processing
# =======================

# Maximum file size in bytes (500MB - handles very large files)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Maximum duration in seconds (1 hour)
MAX_DURATION = 3600

# Supported audio formats (expanded for better compatibility)
SUPPORTED_FORMATS = {
    '.mp3', '.wav', '.m4a', '.flac', '.aac', 
    '.ogg', '.wma', '.webm', '.mp4', '.3gp'
}

# =======================
# Language & Transcription Settings
# =======================

# Primary language for transcription. "hi" for Hindi.
PRIMARY_LANGUAGE = "hi"

# Whisper transcription parameters optimized for COMPLETE audio coverage
WHISPER_OPTIONS = {
    "beam_size": 5,  # Higher beam size for better accuracy
    "temperature": 0.0,  # Deterministic output
    "task": "transcribe",  # vs "translate"
    "vad_filter": False,  # DISABLED - ensures NO audio segments are lost
    "word_timestamps": True,  # Get detailed timing information
    # VAD parameters (not used when vad_filter=False)
    "vad_parameters": {
        "min_silence_duration_ms": 100,  # Very short - prevents dropping speech
        "speech_pad_ms": 30,  # Add padding around detected speech
        "min_speech_duration_ms": 250,  # Minimum speech segment length
        "max_speech_duration_s": 30  # Maximum segment length before splitting
    }
}

# Supported languages for auto-detection
SUPPORTED_LANGUAGES = {
    'hi': 'Hindi', 'en': 'English', 'ur': 'Urdu', 'bn': 'Bengali',
    'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada',
    'gu': 'Gujarati', 'pa': 'Punjabi', 'mr': 'Marathi', 'or': 'Odia',
    'as': 'Assamese', 'ne': 'Nepali', 'zh': 'Chinese', 'ja': 'Japanese',
    'ko': 'Korean', 'ar': 'Arabic', 'fr': 'French', 'de': 'German',
    'es': 'Spanish', 'pt': 'Portuguese', 'it': 'Italian', 'ru': 'Russian'
}

# =======================
# Caching & Directories
# =======================

# Enable or disable transcription result caching
ENABLE_CACHING = True

# Directory for storing cached transcription results
CACHE_DIR = Path("temp/cache")

# Directory for storing uploaded audio files
UPLOAD_DIR = Path("uploads")

# Temporary directory for intermediate files (e.g., preprocessed audio)
TEMP_DIR = Path("temp")

# =======================
# Server Configuration
# =======================

# Server host and port
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.getenv("PORT", "8000"))

# Enable debug mode (not recommended for production)
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Log level (e.g., "INFO", "DEBUG", "WARNING", "ERROR")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =======================
# UI/Frontend Configuration
# =======================

# Application title displayed in the UI
APP_TITLE = "Advanced Audio Transcription - Whisper Powered"

# =======================
# Startup Initialization
# =======================

# Ensure necessary directories exist when the application starts
def initialize_directories():
    for directory in [UPLOAD_DIR, TEMP_DIR, CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Run initialization
initialize_directories()