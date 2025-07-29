# Configuration file for Hindi Audio Transcription Web App
# Advanced settings for performance tuning and customization

import os
from pathlib import Path

# =======================
# Model Configuration
# =======================

# Whisper model size - impacts accuracy vs speed tradeoff
# Options: "tiny", "base", "small", "medium", "large-v3"
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Use faster-whisper if available (recommended)
USE_FASTER_WHISPER = True

# Compute type for faster-whisper
# Options: "int8", "int16", "float16", "float32"
COMPUTE_TYPE = "int8"  # Best for CPU

# Number of CPU threads for processing
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))

# =======================
# Audio Processing
# =======================

# Maximum file size in bytes (500MB default)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Chunk duration for large files (in seconds)
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", "300"))  # 5 minutes

# Supported audio formats
SUPPORTED_FORMATS = {
    '.mp3', '.wav', '.m4a', '.flac', 
    '.aac', '.ogg', '.wma', '.webm'
}

# Audio preprocessing settings
AUDIO_PREPROCESSING = {
    'sample_rate': 16000,      # Optimal for Whisper
    'channels': 1,             # Mono audio
    'bit_depth': 16,          # 16-bit PCM
    'normalize': True,         # Normalize audio levels
    'noise_reduction': True,   # Apply noise filtering
}

# FFmpeg audio filters
FFMPEG_FILTERS = "volume=1.5,highpass=f=200,lowpass=f=3000"

# =======================
# Language Settings
# =======================

# Default language for transcription
DEFAULT_LANGUAGE = "hi"  # Hindi

# Enable automatic language detection
AUTO_LANGUAGE_DETECTION = True

# Language confidence threshold
LANGUAGE_CONFIDENCE_THRESHOLD = 0.7

# Supported languages (if auto-detection is enabled)
SUPPORTED_LANGUAGES = [
    "hi",  # Hindi
    "en",  # English
    "ur",  # Urdu
    "bn",  # Bengali
    "pa",  # Punjabi
    "gu",  # Gujarati
    "mr",  # Marathi
    "ta",  # Tamil
    "te",  # Telugu
    "kn",  # Kannada
    "ml",  # Malayalam
    "or",  # Odia
    "as",  # Assamese
]

# =======================
# Performance Settings
# =======================

# Enable result caching
ENABLE_CACHING = True

# Cache directory
CACHE_DIR = Path("temp/cache")

# Cache expiry (in seconds) - 7 days default
CACHE_EXPIRY = 7 * 24 * 60 * 60

# Maximum concurrent transcriptions
MAX_CONCURRENT_TRANSCRIPTIONS = 2

# Memory usage limit (in MB)
MEMORY_LIMIT = 2048

# =======================
# Server Configuration
# =======================

# Server host and port
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.getenv("PORT", "8000"))

# Enable debug mode
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Upload directory
UPLOAD_DIR = Path("uploads")

# Temporary directory
TEMP_DIR = Path("temp")

# Log directory
LOG_DIR = Path("logs")

# =======================
# UI Configuration
# =======================

# Application title
APP_TITLE = "Hindi Audio Transcription"

# Theme colors
THEME_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'info': '#3b82f6'
}

# Progress update interval (in milliseconds)
PROGRESS_UPDATE_INTERVAL = 500

# Toast notification duration (in milliseconds)
TOAST_DURATION = 5000

# =======================
# Advanced Settings
# =======================

# Whisper model parameters
WHISPER_PARAMS = {
    'temperature': 0.0,           # Deterministic output
    'beam_size': 5,              # Beam search size
    'best_of': 5,                # Number of candidates
    'condition_on_previous_text': True,  # Use context
    'compression_ratio_threshold': 2.4,  # Compression detection
    'logprob_threshold': -1.0,   # Log probability threshold
    'no_speech_threshold': 0.6,  # Silence detection
    'word_timestamps': False,    # Enable word-level timestamps
}

# Voice Activity Detection (VAD) settings
VAD_SETTINGS = {
    'enable': True,
    'min_silence_duration_ms': 500,
    'max_speech_duration_s': 30,
    'speech_pad_ms': 30
}

# Batch processing settings
BATCH_PROCESSING = {
    'enable': False,              # Enable batch mode
    'max_batch_size': 5,         # Maximum files per batch
    'batch_timeout': 300,        # Timeout in seconds
}

# =======================
# Error Handling
# =======================

# Retry settings
RETRY_SETTINGS = {
    'max_retries': 3,
    'retry_delay': 1.0,          # Initial delay in seconds
    'backoff_multiplier': 2.0,   # Exponential backoff
}

# Error messages
ERROR_MESSAGES = {
    'file_too_large': 'File size exceeds the maximum limit of 500MB',
    'unsupported_format': 'Unsupported audio format. Please use MP3, WAV, M4A, FLAC, AAC, OGG, or WMA',
    'model_loading_failed': 'Failed to load the AI model. Please try again',
    'transcription_failed': 'Transcription failed. Please check your audio file and try again',
    'ffmpeg_not_found': 'FFmpeg is required but not found. Please install FFmpeg',
    'insufficient_memory': 'Insufficient memory. Try using a smaller model or file',
}

# =======================
# Security Settings
# =======================

# File upload restrictions
UPLOAD_RESTRICTIONS = {
    'max_filename_length': 255,
    'allowed_extensions_only': True,
    'scan_for_malware': False,    # Requires additional setup
}

# Rate limiting
RATE_LIMITING = {
    'enable': False,              # Enable rate limiting
    'requests_per_minute': 10,   # Max requests per minute per IP
    'requests_per_hour': 100,    # Max requests per hour per IP
}

# CORS settings
CORS_SETTINGS = {
    'allow_origins': ["*"],
    'allow_credentials': True,
    'allow_methods': ["*"],
    'allow_headers': ["*"],
}

# =======================
# Monitoring & Analytics
# =======================

# Enable metrics collection
ENABLE_METRICS = False

# Metrics to collect
METRICS = {
    'transcription_count': True,
    'processing_time': True,
    'file_sizes': True,
    'error_rates': True,
    'model_usage': True,
}

# Health check settings
HEALTH_CHECK = {
    'enable': True,
    'check_model': True,
    'check_ffmpeg': True,
    'check_disk_space': True,
    'min_disk_space_mb': 1024,   # Minimum free space in MB
}

# =======================
# Development Settings
# =======================

# Enable development mode features
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

# Auto-reload on code changes
AUTO_RELOAD = DEV_MODE

# Enable API documentation
ENABLE_API_DOCS = True

# API documentation URL
API_DOCS_URL = "/docs" if ENABLE_API_DOCS else None

# Enable interactive API exploration
ENABLE_REDOC = True
REDOC_URL = "/redoc" if ENABLE_REDOC else None
