#!/bin/bash

# Hindi Audio Transcription Web App Run Script
# This script starts the FastAPI server

echo "🎤 Starting Hindi Audio Transcription Web App..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required directories exist
mkdir -p uploads temp/cache temp/chunks logs

# Set environment variables for optimal performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Start the server
echo "🚀 Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the FastAPI app with uvicorn
python backend/app.py
