# 🎤 Hindi Audio Transcription Web App

A complete local web application for transcribing Hindi audio files into accurate text using AI models. Built with FastAPI, OpenAI Whisper, and modern web technologies.

![Demo Screenshot](demo-screenshot.png)

## ✨ Features

### 🧠 AI-Powered Transcription
- **OpenAI Whisper Integration**: Uses faster-whisper for optimal performance
- **Hindi Language Optimized**: Specifically tuned for Hindi speech recognition
- **Multiple Model Support**: Automatic selection between base, medium, and large models
- **Auto Language Detection**: Defaults to Hindi but can detect other languages

### 🖥️ Modern Web Interface
- **Drag & Drop Upload**: Easy file upload with visual feedback
- **Real-time Progress**: WebSocket-powered progress tracking
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Professional UI**: Clean, modern interface with smooth animations

### ⚡ Performance Optimized
- **Memory Efficient**: Automatic audio chunking for large files
- **CPU Optimized**: Uses faster-whisper and CPU-optimized PyTorch
- **Smart Caching**: Avoids reprocessing the same files
- **Background Processing**: Non-blocking transcription with progress updates

### 🛠️ Audio Processing
- **FFmpeg Integration**: Automatic audio normalization and conversion
- **Multiple Formats**: MP3, WAV, M4A, FLAC, AAC, OGG, WMA support
- **Preprocessing**: Voice activity detection and audio filtering
- **Large File Support**: Handles files up to 500MB with chunking

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- FFmpeg (automatically installed on macOS/Linux)
- 4GB+ RAM (8GB recommended for large files)

### Installation

1. **Clone or Download** this project to your local machine

2. **Run the installation script**:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Start the application**:
   ```bash
   ./run.sh
   ```

4. **Open your browser** and go to: `http://localhost:8000`

That's it! The application is now ready to use.

## 📖 Usage Guide

### Step 1: Upload Audio File
- **Drag & Drop**: Simply drag your audio file onto the upload area
- **Browse**: Click the upload area to select a file from your computer
- **Supported Formats**: MP3, WAV, M4A, FLAC, AAC, OGG, WMA (max 500MB)

### Step 2: Start Transcription
- Click the "Start Transcription" button
- Watch the real-time progress updates
- Processing time varies based on file length and your hardware

### Step 3: View Results
- **Read Transcription**: View the complete transcribed text
- **Copy Text**: One-click copy to clipboard
- **Download**: Save transcription as a text file
- **Statistics**: View word count, model used, and processing time

### Step 4: New File
- Click "New File" to transcribe another audio file
- Previous results are cached for faster access

## 🔧 Advanced Configuration

### Model Selection
Edit `backend/app.py` to change the Whisper model:
```python
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v3
```

**Model Comparison**:
- `tiny`: Fastest, least accurate, ~39MB
- `base`: Good balance, ~74MB (default)
- `small`: Better accuracy, ~244MB
- `medium`: High accuracy, ~769MB
- `large-v3`: Best accuracy, ~1550MB

### Chunk Duration
For very long files, adjust chunk processing:
```python
CHUNK_DURATION = 300  # seconds (default: 5 minutes)
```

### Port Configuration
Change the server port in `backend/app.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change port here
```

## 📁 Project Structure

```
audiototextusingwarp/
├── backend/
│   └── app.py              # Main FastAPI application
├── static/
│   ├── css/
│   │   └── style.css       # UI styling
│   └── js/
│       └── main.js         # Frontend JavaScript
├── templates/
│   └── index.html          # Main HTML template
├── uploads/                # Uploaded audio files
├── temp/
│   ├── cache/             # Transcription cache
│   └── chunks/            # Audio chunks for processing
├── requirements.txt        # Python dependencies
├── install.sh             # Installation script
├── run.sh                 # Application launcher
└── README.md              # This file
```

## 🔍 API Endpoints

### Upload File
```
POST /upload
Content-Type: multipart/form-data
```

### Start Transcription
```
POST /transcribe/{file_id}
```

### WebSocket Progress
```
WS /ws/{client_id}
```

### Health Check
```
GET /health
```

## 🐛 Troubleshooting

### Common Issues

**1. "FFmpeg not found" error**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

**2. "Model loading failed" error**
- Ensure you have sufficient RAM (4GB+ recommended)
- Try using a smaller model (tiny or base)
- Check internet connection for initial model download

**3. "File too large" error**
- The app supports files up to 500MB
- For larger files, consider splitting them first
- Ensure sufficient disk space in the temp directory

**4. Slow transcription**
- Use CPU-optimized build (installed by default)
- Close other resource-intensive applications
- Consider using a smaller Whisper model

### Performance Tips

1. **Optimal Audio Format**: Use 16kHz mono WAV files for best performance
2. **File Size**: Keep files under 100MB for faster processing
3. **Hardware**: More CPU cores = faster transcription
4. **Memory**: 8GB+ RAM recommended for large files

## 🛡️ Security & Privacy

- **Local Processing**: All transcription happens on your machine
- **No Data Upload**: Audio files are processed locally, never sent to external servers
- **Temporary Files**: Automatically cleaned up after processing
- **Cache Management**: Transcription cache can be cleared manually

## 🤝 Contributing

This project is designed to be self-contained and easy to modify. Feel free to:

- Improve the UI/UX design
- Add new audio format support
- Optimize transcription performance
- Add new features like speaker identification
- Improve Hindi language accuracy

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenAI Whisper**: For the excellent speech recognition model
- **FastAPI**: For the modern Python web framework
- **faster-whisper**: For the optimized Whisper implementation
- **FFmpeg**: For audio processing capabilities

## 📞 Support

If you encounter any issues:

1. Check the console logs in your browser (F12 → Console)
2. Review the server logs in the terminal
3. Ensure your audio file is in a supported format
4. Verify FFmpeg is properly installed
5. Try with a smaller audio file first

---

**Enjoy transcribing your Hindi audio files! 🎉**
