#!/bin/bash

# Hindi Audio Transcription Web App Installation Script
# This script sets up the environment and installs all dependencies

set -e  # Exit on any error

echo "🎤 Hindi Audio Transcription Web App Setup"
echo "=========================================="

# Check if Python 3.8+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION found, but version $REQUIRED_VERSION or higher is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Installing..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ Homebrew not found. Please install FFmpeg manually:"
            echo "   Visit: https://ffmpeg.org/download.html"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        else
            echo "❌ Package manager not supported. Please install FFmpeg manually."
            exit 1
        fi
    else
        echo "❌ Unsupported OS. Please install FFmpeg manually."
        exit 1
    fi
fi

echo "✅ FFmpeg is available"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CPU support (smaller footprint)
echo "🔥 Installing PyTorch (CPU version for faster startup)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads temp/cache temp/chunks logs

# Download the Whisper base model (will be cached)
echo "🤖 Pre-downloading Whisper base model..."
python3 -c "
import whisper
print('Downloading Whisper base model...')
model = whisper.load_model('base')
print('Model downloaded and cached successfully!')
"

# Set permissions
chmod +x run.sh

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Run the application: ./run.sh"
echo "2. Open your browser and go to: http://localhost:8000"
echo "3. Upload a Hindi audio file and test the transcription"
echo ""
echo "💡 Tips:"
echo "- For better performance, use audio files under 100MB"
echo "- Supported formats: MP3, WAV, M4A, FLAC, AAC, OGG, WMA"
echo "- The first transcription may take longer due to model loading"
echo ""
echo "🛠️  If you encounter issues:"
echo "- Check the logs in the logs/ directory"
echo "- Ensure your audio file is in a supported format"
echo "- For very long files, the app will automatically chunk them"
echo ""
