import os
import asyncio
import logging
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import aiofiles
import ffmpeg
import torch
import subprocess

# Import OpenAI Whisper for best Hindi accuracy
import whisper
WHISPER_AVAILABLE = True

# Try to import faster variants as backup
try:
    from whisper_ctranslate2 import WhisperCT2
    WHISPER_CT2_AVAILABLE = True
except ImportError:
    WHISPER_CT2_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Hindi Audio Transcription", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
whisper_model = None
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v3
CHUNK_DURATION = 300  # 5 minutes per chunk to avoid memory issues
SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
CACHE_DIR = Path("temp/cache")
CACHE_DIR.mkdir(exist_ok=True)

class ConnectionManager:
    """Manage WebSocket connections for real-time progress updates"""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_progress(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send progress to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

def clean_hindi_text(text: str) -> str:
    """Clean and ensure proper Hindi text output"""
    if not text:
        return ""
    
    # Remove repeated characters more aggressively (like ششششش... or ।।।।...)
    text = re.sub(r'(.)\1{3,}', r'\1', text)  # Changed from 5+ to 3+ repetitions
    
    # Remove specific repeated punctuation patterns
    text = re.sub(r'[।]{3,}', '।', text)  # Clean repeated Hindi periods
    text = re.sub(r'[\s]{3,}', ' ', text)  # Clean multiple spaces
    
    # Remove or replace common encoding artifacts
    text = text.replace('�', '')
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def ensure_hindi_script(text: str) -> str:
    """Ensure the text is in proper Hindi Devanagari script"""
    if not text:
        return ""
    
    # First clean the text for any artifacts
    text = clean_hindi_text(text)
    
    # Check if text contains Arabic/Urdu script characters
    arabic_pattern = r'[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]'
    
    if re.search(arabic_pattern, text):
        logger.info("Detected Urdu/Arabic script, converting to Hindi Devanagari")
        
        # Try multiple approaches to convert Urdu to Hindi
        try:
            # Method 1: Use indic-transliteration (more reliable for Urdu to Hindi)
            from indic_transliteration import sanscript
            hindi_text = sanscript.transliterate(text, sanscript.URDU, sanscript.DEVANAGARI)
            logger.info(f"Successfully converted using indic-transliteration: {hindi_text[:100]}...")
            return hindi_text
        except Exception as e:
            logger.warning(f"indic-transliteration failed: {e}")
        
        try:
            # Method 2: Use transliterate library (fallback)
            from transliterate import translit
            # Try different language codes for Urdu to Hindi conversion
            hindi_text = translit(text, 'ur', reversed=False)  # Urdu to Latin
            hindi_text = translit(hindi_text, 'hi')  # Latin to Hindi
            logger.info(f"Successfully converted using transliterate: {hindi_text[:100]}...")
            return hindi_text
        except Exception as e:
            logger.warning(f"transliterate library failed: {e}")
        
        # Method 3: Manual character mapping for common Urdu to Hindi conversions
        urdu_to_hindi_map = {
            'ا': 'अ', 'ب': 'ब', 'پ': 'प', 'ت': 'त', 'ٹ': 'ट', 'ث': 'स', 'ج': 'ज', 'چ': 'च', 'ح': 'ह', 'خ': 'ख',
            'د': 'द', 'ڈ': 'ड', 'ذ': 'ज़', 'ر': 'र', 'ڑ': 'ड़', 'ز': 'ज़', 'ژ': 'ज़', 'س': 'स', 'ش': 'श', 'ص': 'स',
            'ض': 'ज़', 'ط': 'त', 'ظ': 'ज़', 'ع': 'अ', 'غ': 'ग़', 'ف': 'फ', 'ق': 'क़', 'ک': 'क', 'گ': 'ग', 'ل': 'ल',
            'م': 'म', 'ن': 'न', 'ں': 'ं', 'و': 'व', 'ہ': 'ह', 'ھ': 'ह', 'ء': 'अ', 'ی': 'ी', 'ے': 'े', 'ً': 'ं',
            'ُ': 'ु', 'ِ': 'ि', 'َ': 'ा', 'ْ': '', '۔': '।', ' ': ' '
        }
        
        # Apply manual mapping
        for urdu_char, hindi_char in urdu_to_hindi_map.items():
            text = text.replace(urdu_char, hindi_char)
        
        return text
    
    return text

def load_whisper_model() -> Any:
    """Load the most accurate Whisper model available"""
    global whisper_model
    
    if whisper_model is None:
        try:
            # Prioritize OpenAI Whisper for best accuracy
            logger.info(f"Loading OpenAI Whisper model: {MODEL_SIZE}")
            whisper_model = whisper.load_model(MODEL_SIZE)
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    return whisper_model

def get_file_hash(file_path: str) -> str:
    """Generate hash for file caching"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def preprocess_audio(input_path: str, output_path: str) -> bool:
    """Preprocess audio for better transcription results"""
    try:
        logger.info(f"Preprocessing audio: {input_path}")
        
        # Use simpler FFmpeg processing to avoid artifacts
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                acodec='pcm_s16le',  # 16-bit PCM
                ac=1,                # Convert to mono
                ar=16000             # 16kHz sample rate (optimal for Whisper)
                # Removed aggressive filtering that might cause artifacts
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True)
        )
        
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg preprocessing failed: {e}")
        # Simple fallback - just copy the file
        try:
            shutil.copy2(input_path, output_path)
            return True
        except Exception as fallback_e:
            logger.error(f"File copy fallback failed: {fallback_e}")
            return False

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffmpeg"""
    try:
        probe = ffmpeg.probe(audio_path)
        return float(probe['streams'][0]['duration'])
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0

def chunk_audio(audio_path: str, chunk_duration: int = CHUNK_DURATION) -> list:
    """Split long audio files into manageable chunks using ffmpeg"""
    try:
        duration = get_audio_duration(audio_path)
        
        if duration <= chunk_duration:
            return [audio_path]
        
        chunks = []
        chunk_dir = Path("temp/chunks")
        chunk_dir.mkdir(exist_ok=True)
        
        num_chunks = int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = chunk_dir / f"chunk_{i:03d}.wav"
            
            # Use ffmpeg to extract chunk
            (
                ffmpeg
                .input(audio_path, ss=start_time, t=chunk_duration)
                .output(str(chunk_path), acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            
            chunks.append(str(chunk_path))
        
        return chunks
    except Exception as e:
        logger.error(f"Audio chunking failed: {e}")
        return [audio_path]

async def transcribe_audio_chunk(chunk_path: str, model: Any, client_id: str = None) -> str:
    """Transcribe a single audio chunk"""
    try:
        logger.info(f"Starting transcription for chunk: {chunk_path}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
        
        # Check if chunk file exists and has content
        if not os.path.exists(chunk_path):
            logger.error(f"Chunk file does not exist: {chunk_path}")
            return ""
        
        file_size = os.path.getsize(chunk_path)
        logger.info(f"Chunk file size: {file_size} bytes")
        
        if file_size == 0:
            logger.warning(f"Chunk file is empty: {chunk_path}")
            return ""
        
        # Determine which model type we're using based on the model object
        if hasattr(model, 'transcribe') and hasattr(model, 'model'):
            logger.info("Using whisper-ctranslate2 model")
            # This is likely whisper-ctranslate2
            result = model.transcribe(
                chunk_path,
                language="hi",  # Force Hindi language
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,  # Disable to avoid Urdu influence
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(min_silence_duration_ms=500),
                task="transcribe"  # Explicitly set transcription task
            )
            
            logger.info(f"whisper-ctranslate2 result type: {type(result)}")
            logger.info(f"whisper-ctranslate2 result: {result}")
            
            # Extract text from whisper-ctranslate2 result
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
            elif hasattr(result, 'text'):
                text = result.text
            else:
                # Handle segments format
                text = " ".join([segment.text for segment in result if hasattr(segment, 'text')])
                
        elif hasattr(model, 'transcribe') and hasattr(model, 'feature_extractor'):
            logger.info("Using faster-whisper model")
            # This is likely faster-whisper
            segments, info = model.transcribe(
                chunk_path,
                language="hi",  # Force Hindi language
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,  # Enable for better continuity
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(min_silence_duration_ms=500),
                task="transcribe"  # Explicitly set transcription task
            )
            
            logger.info(f"faster-whisper segments type: {type(segments)}")
            text = " ".join([segment.text for segment in segments])
            logger.info(f"faster-whisper extracted text: {text}")
        else:
            logger.info("Using OpenAI Whisper model")
            # Use OpenAI Whisper
            result = model.transcribe(
                chunk_path,
                language="hi",  # Force Hindi
                temperature=0.0,
                condition_on_previous_text=True,
                task="transcribe"  # Explicitly set transcription task
            )
            
            logger.info(f"OpenAI Whisper result type: {type(result)}")
            logger.info(f"OpenAI Whisper result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
                logger.info(f"OpenAI Whisper extracted text: '{text}'")
            else:
                logger.error(f"Unexpected result format from OpenAI Whisper: {result}")
                text = ""
        
        logger.info(f"Final transcribed text: '{text.strip()}'")
        return text.strip()
    
    except Exception as e:
        logger.error(f"Transcription failed for chunk {chunk_path}: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ""

async def transcribe_full_audio(file_path: str, client_id: str = None) -> Dict[str, Any]:
    """Main transcription function with progress tracking"""
    try:
        # Check cache first
        file_hash = get_file_hash(file_path)
        cache_file = CACHE_DIR / f"{file_hash}.json"
        
        if cache_file.exists():
            logger.info("Using cached transcription")
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                cached_result = json.loads(await f.read())
                if client_id:
                    await manager.send_progress(client_id, {
                        "status": "completed",
                        "progress": 100,
                        "message": "Transcription completed (from cache)"
                    })
                return cached_result
        
        # Load model
        if client_id:
            await manager.send_progress(client_id, {
                "status": "loading_model",
                "progress": 5,
                "message": "Loading AI model..."
            })
        
        model = load_whisper_model()
        
        # Preprocess audio
        if client_id:
            await manager.send_progress(client_id, {
                "status": "preprocessing",
                "progress": 15,
                "message": "Preprocessing audio..."
            })
        
        processed_path = Path("temp") / f"processed_{Path(file_path).stem}.wav"
        if not await preprocess_audio(file_path, str(processed_path)):
            processed_path = Path(file_path)
        
        # Chunk audio if necessary
        if client_id:
            await manager.send_progress(client_id, {
                "status": "chunking",
                "progress": 25,
                "message": "Analyzing audio length..."
            })
        
        chunks = chunk_audio(str(processed_path))
        total_chunks = len(chunks)
        
        # Transcribe chunks
        transcribed_texts = []
        for i, chunk_path in enumerate(chunks):
            if client_id:
                progress = 30 + (i / total_chunks) * 60
                await manager.send_progress(client_id, {
                    "status": "transcribing",
                    "progress": int(progress),
                    "message": f"Transcribing part {i+1} of {total_chunks}..."
                })
            
            chunk_text = await transcribe_audio_chunk(chunk_path, model, client_id)
            if chunk_text:
                # First ensure it's in Hindi Devanagari script
                hindi_text = ensure_hindi_script(chunk_text)
                # Then clean repeated characters and artifacts
                cleaned_text = clean_hindi_text(hindi_text)
                if cleaned_text.strip():
                    transcribed_texts.append(cleaned_text)
            
            # Clean up chunk files
            try:
                if chunk_path != str(processed_path):
                    os.unlink(chunk_path)
            except:
                pass
        
        # Combine results
        full_text = " ".join(transcribed_texts)
        
        # Prepare result
        result = {
            "text": full_text,
            "language": "hi",
            "timestamp": datetime.now().isoformat(),
            "chunks_processed": total_chunks,
            "model_used": MODEL_SIZE,
            "word_count": len(full_text.split()) if full_text else 0
        }
        
        # Cache result
        async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(result, ensure_ascii=False, indent=2))
        
        # Clean up processed file
        try:
            if processed_path != Path(file_path):
                os.unlink(processed_path)
        except:
            pass
        
        if client_id:
            await manager.send_progress(client_id, {
                "status": "completed",
                "progress": 100,
                "message": "Transcription completed successfully!"
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        if client_id:
            await manager.send_progress(client_id, {
                "status": "error",
                "progress": 0,
                "message": f"Transcription failed: {str(e)}"
            })
        raise HTTPException(status_code=500, detail=str(e))

# API Routes
@app.get("/")
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and validate audio file"""
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Generate unique filename
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Validate file size (max 500MB)
        file_size = len(content)
        if file_size > 500 * 1024 * 1024:
            os.unlink(file_path)
            raise HTTPException(status_code=400, detail="File too large (max 500MB)")
        
        return JSONResponse({
            "success": True,
            "file_id": safe_filename,
            "file_size": file_size,
            "message": "File uploaded successfully"
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/{file_id}")
async def start_transcription(file_id: str):
    """Start transcription process"""
    try:
        file_path = Path("uploads") / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Start transcription in background
        result = await transcribe_full_audio(str(file_path))
        
        return JSONResponse({
            "success": True,
            "transcription": result,
            "message": "Transcription completed"
        })
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for real-time progress updates"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": whisper_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
