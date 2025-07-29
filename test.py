#!/usr/bin/env python3
"""
Test script to verify the Hindi Audio Transcription Web App installation
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} found, but 3.8+ is required")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'whisper',
        'ffmpeg',
        'pydub',
        'aiofiles'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'ffmpeg':
                import ffmpeg
            else:
                importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_ffmpeg():
    """Check if FFmpeg is available in system PATH"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg is available: {version_line}")
            return True
        else:
            print("❌ FFmpeg is not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg check timed out")
        return False

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        'backend',
        'static/css',
        'static/js',
        'templates',
        'uploads',
        'temp/cache',
        'temp/chunks'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ Directory {dir_path} exists")
        else:
            print(f"❌ Directory {dir_path} is missing")
            all_exist = False
    
    return all_exist

def check_files():
    """Check if required files exist"""
    required_files = [
        'backend/app.py',
        'static/css/style.css',
        'static/js/main.js',
        'templates/index.html',
        'requirements.txt',
        'install.sh',
        'run.sh',
        'README.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} is missing")
            all_exist = False
    
    return all_exist

def test_whisper_model():
    """Test if Whisper model can be loaded"""
    try:
        import whisper
        print("🤖 Testing Whisper model loading...")
        model = whisper.load_model("base")
        print("✅ Whisper base model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Hindi Audio Transcription Web App Installation")
    print("=" * 60)
    
    tests = [
        ("Python Version", check_python_version),
        ("Required Directories", check_directories),
        ("Required Files", check_files),
        ("FFmpeg Installation", check_ffmpeg),
        ("Python Dependencies", lambda: check_dependencies()[0]),
        ("Whisper Model", test_whisper_model),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ⚠️  {test_name} test failed")
        except Exception as e:
            print(f"   ❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your installation is ready.")
        print("🚀 Run './run.sh' to start the application")
        return True
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        print("💡 Try running './install.sh' again")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
