# Configuration file for Hindi Audio Transcription Web App

import os
from pathlib import Path

# =======================
# Model Configuration
# =======================

# Whisper model size - impacts accuracy vs speed tradeoff
# Options: "tiny", "base", "small", "medium", "large-v3"
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")



# =======================
# Audio Processing
# =======================

# Maximum file size in bytes (e.g., 500 * 1024 * 1024 for 500MB)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Supported audio formats
SUPPORTED_FORMATS = {
    '.mp3', '.wav', '.m4a', '.flac', 
    '.aac', '.ogg', '.wma', '.webm'
}

# =======================
# Language & Transcription Settings
# =======================

# Default language for transcription. "hi" for Hindi.
DEFAULT_LANGUAGE = "hi"

# faster-whisper transcription parameters
# These are fine-tuned for better accuracy with Hindi.
TRANSCRIPTION_OPTIONS = {
    "beam_size": 5,
    "temperature": 0.0,  # Set to 0.0 for deterministic, consistent output
    "condition_on_previous_text": False,  # Prevents repetitive loops and language drift
    "vad_filter": True,  # Enable Voice Activity Detection to filter out silence
    "vad_parameters": {"min_silence_duration_ms": 500} # VAD setting
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
APP_TITLE = "Hindi Audio Transcription Service"

# =======================
# Startup Initialization
# =======================

# Ensure necessary directories exist when the application starts
def initialize_directories():
    for directory in [UPLOAD_DIR, TEMP_DIR, CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Run initialization
initialize_directories()