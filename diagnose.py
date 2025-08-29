#!/usr/bin/env python3
"""
Diagnostic script to identify server startup issues
"""

import sys
print(f"Python version: {sys.version}")

# Test basic imports
try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    import uvicorn
    print("✅ Uvicorn imported successfully")
except ImportError as e:
    print(f"❌ Uvicorn import failed: {e}")

try:
    import whisper
    print("✅ Whisper imported successfully")
except ImportError as e:
    print(f"❌ Whisper import failed: {e}")

try:
    from faster_whisper import WhisperModel
    print("✅ Faster-whisper imported successfully")
except ImportError as e:
    print(f"❌ Faster-whisper import failed: {e}")

try:
    import torch
    print(f"✅ PyTorch imported successfully - version: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import librosa
    print("✅ Librosa imported successfully")
except ImportError as e:
    print(f"❌ Librosa import failed: {e}")

try:
    import noisereduce
    print("✅ Noisereduce imported successfully")
except ImportError as e:
    print(f"❌ Noisereduce import failed: {e}")

try:
    from langdetect import detect
    print("✅ Langdetect imported successfully")
except ImportError as e:
    print(f"❌ Langdetect import failed: {e}")

# Test backend app import
print("\n--- Testing backend app import ---")
try:
    import backend.app
    print("✅ Backend app imported successfully")
except Exception as e:
    print(f"❌ Backend app import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test complete ---")