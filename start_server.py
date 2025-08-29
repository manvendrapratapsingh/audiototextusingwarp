#!/usr/bin/env python3
"""
Simple server startup script with error handling
"""

import os
import sys
import subprocess

def start_server():
    try:
        # Change to project directory
        os.chdir('/Users/manvendrapratapsingh/Documents/audiototextusingwarp')
        
        # Activate virtual environment and start server
        cmd = [
            'bash', '-c',
            'source venv/bin/activate && python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload'
        ]
        
        print("🚀 Starting audio transcription server...")
        print("📍 Project directory: /Users/manvendrapratapsingh/Documents/audiototextusingwarp")
        print("🌐 Server will be available at: http://localhost:8000")
        print("⏳ Please wait while the server starts...")
        
        # Run the command
        process = subprocess.run(cmd, capture_output=False, text=True)
        
        if process.returncode != 0:
            print(f"❌ Server failed to start with return code: {process.returncode}")
        else:
            print("✅ Server started successfully!")
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_server()