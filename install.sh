#!/bin/bash

# Advanced Audio Transcription - Installation Script
# Powered by OpenAI Whisper for exceptional Hindi accuracy

set -e  # Exit on any error

echo "🚀 Installing Advanced Audio Transcription Service..."
echo "📦 This will install OpenAI Whisper for superior Hindi transcription"
echo "=============================================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Verify Python version is 3.8+
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Please install FFmpeg:"
    echo "   macOS: brew install ffmpeg"
    echo "   Ubuntu: sudo apt install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ FFmpeg is available"
fi

# Create virtual environment
echo "📁 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
echo "   This may take a few minutes as it downloads AI models..."
pip install -r requirements.txt

# Pre-download Whisper model (optional)
echo "🤖 Pre-downloading Whisper model for faster first run..."
python3 -c "
try:
    import whisper
    print('🔄 Downloading Whisper large-v3 model...')
    whisper.load_model('large-v3')
    print('✅ Model downloaded successfully!')
except Exception as e:
    print('⚠️  Model will be downloaded on first use')
    print(f'   Error: {e}')
" 2>/dev/null || echo "   Model will be downloaded on first use"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads temp/cache temp/chunks logs

# Set permissions
chmod +x run.sh

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "🎯 Key Features:"
echo "   • OpenAI Whisper large-v3 model for exceptional Hindi accuracy"
echo "   • Faster-whisper for 4x speed improvement"
echo "   • Support for 95+ languages with auto-detection"
echo "   • Audio enhancement and noise reduction"
echo "   • Supports files up to 500MB"
echo ""
echo "🚀 To start the application:"
echo "   ./run.sh"
echo ""
echo "🌐 Then open: http://localhost:8000"
echo ""
echo "💡 Tips:"
echo "   • For GPU acceleration (10x faster), install CUDA and uncomment GPU dependencies in requirements.txt"
echo "   • Large model provides best Hindi accuracy but requires ~3GB RAM"
echo "   • Use 'medium' model in config.py for faster processing on low-memory systems"
echo "   • First transcription may take longer due to model loading"
echo ""
echo "🛠️  Troubleshooting:"
echo "   • Check logs/ directory for error details"
echo "   • Ensure audio files are in supported formats"
echo "   • For issues, check: https://github.com/openai/whisper"