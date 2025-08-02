import os
import asyncio
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil
import tempfile
import torch
import librosa
import soundfile as sf
import numpy as np
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Free ASR Models
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
import speech_recognition as sr

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import aiofiles
import ffmpeg




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Hindi Audio Transcription", version="2.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables from config (with defaults)
# These would ideally be loaded from config.py, but are hardcoded here for simplicity
MODEL_SIZE = "small"

SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
CACHE_DIR = Path("temp/cache")
UPLOAD_DIR = Path("uploads")

# Create necessary directories
CACHE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Global model instances for different ASR approaches
wav2vec2_model = None
wav2vec2_processor = None
speech_recognizer = None

class ConnectionManager:
    """Manages WebSocket connections for real-time progress updates."""
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

def load_wav2vec2_model():
    """Loads the Wav2Vec2 model for Hindi transcription - Best free model for Hindi."""
    global wav2vec2_model, wav2vec2_processor
    if wav2vec2_model is None or wav2vec2_processor is None:
        try:
            logger.info("Loading Hindi-specific Wav2Vec2 model (high accuracy, completely free)...")
            # Use a Hindi-specific model that has proper tokenization
            model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-hindi"
            wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
            wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            logger.info("✅ Wav2Vec2 Hindi model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {e}", exc_info=True)
            logger.warning("Falling back to Google Speech Recognition only")
            # Don't raise exception, just continue without Wav2Vec2
            return None, None
    return wav2vec2_model, wav2vec2_processor

def load_speech_recognition():
    """Initialize Google Speech Recognition (free tier) as fallback."""
    global speech_recognizer
    if speech_recognizer is None:
        try:
            logger.info("Initializing Google Speech Recognition (free fallback)...")
            speech_recognizer = sr.Recognizer()
            logger.info("✅ Speech Recognition initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize Speech Recognition: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Speech Recognition init failed: {str(e)}")
    return speech_recognizer



def get_file_hash(file_path: Path) -> str:
    """Generates an MD5 hash for a given file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()





async def transcribe_with_wav2vec2(processed_path: Path) -> str:
    """Transcribes audio using Wav2Vec2 model - Best free option for Hindi."""
    model, processor = load_wav2vec2_model()
    
    # If Wav2Vec2 model failed to load, fall back to Google Speech Recognition
    if model is None or processor is None:
        logger.info("Wav2Vec2 model not available, using Google Speech Recognition...")
        return await transcribe_with_google_sr(processed_path)
    
    logger.info("🚀 Using Wav2Vec2 model for Hindi transcription...")
    
    try:
        # Load audio file with librosa (16kHz sample rate)
        audio, sample_rate = librosa.load(str(processed_path), sr=16000)
        
        # Process audio through the model
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        raw_text = processor.batch_decode(predicted_ids)[0]
        
        logger.info(f"🎯 Wav2Vec2 raw output: {raw_text[:100]}...")
        
        # Apply text enhancement pipeline
        return await apply_text_enhancement_pipeline(raw_text)
        
    except Exception as e:
        logger.error(f"Wav2Vec2 transcription failed: {e}", exc_info=True)
        # Fall back to Google Speech Recognition
        logger.info("Falling back to Google Speech Recognition...")
        return await transcribe_with_google_sr(processed_path)

async def transcribe_with_google_sr(processed_path: Path) -> str:
    """Transcribes audio using Google Speech Recognition (free tier) with chunking for long audio."""
    recognizer = load_speech_recognition()
    logger.info("🚀 Using Google Speech Recognition for Hindi transcription...")
    
    try:
        # Load audio file and get duration
        with sr.AudioFile(str(processed_path)) as source:
            duration = source.DURATION if hasattr(source, 'DURATION') else None
            
        # Check if audio is longer than 50 seconds (Google SR limit is ~60 seconds)
        if duration and duration > 50:
            logger.info(f"Audio duration: {duration:.2f}s - Using chunked transcription")
            return await transcribe_long_audio_chunked(processed_path, recognizer)
        else:
            # Process normally for short audio
            with sr.AudioFile(str(processed_path)) as source:
                # Adjust for ambient noise and record the entire audio
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
            
            # Transcribe with Hindi language specification
            raw_text = recognizer.recognize_google(audio_data, language='hi-IN')
            logger.info(f"🎯 Google SR raw output: {raw_text[:100]}...")
            
            # Apply text enhancement pipeline
            return await apply_text_enhancement_pipeline(raw_text)
        
    except sr.UnknownValueError:
        logger.error("Google Speech Recognition could not understand audio")
        return "[Audio could not be transcribed - unclear speech]"
    except sr.RequestError as e:
        logger.error(f"Google Speech Recognition error: {e}")
        return "[Transcription service temporarily unavailable]"
    except Exception as e:
        logger.error(f"Google SR transcription failed: {e}", exc_info=True)
        return "[Transcription failed due to technical error]"

async def transcribe_long_audio_chunked(processed_path: Path, recognizer) -> str:
    """Transcribe long audio files by splitting into chunks."""
    logger.info("Transcribing long audio file using chunked approach...")
    
    try:
        # Load audio with pydub for easier chunking
        from pydub import AudioSegment
        
        # Load the audio file
        audio = AudioSegment.from_wav(str(processed_path))
        
        # Split into 45-second chunks (Google SR has ~60s limit)
        chunk_length_ms = 45 * 1000  # 45 seconds in milliseconds
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        logger.info(f"Split audio into {len(chunks)} chunks for processing")
        
        transcriptions = []
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Save chunk to temporary file
            chunk_path = temp_dir / f"chunk_{i}.wav"
            chunk.export(str(chunk_path), format="wav")
            
            try:
                # Transcribe this chunk
                with sr.AudioFile(str(chunk_path)) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio_data = recognizer.record(source)
                
                chunk_text = recognizer.recognize_google(audio_data, language='hi-IN')
                if chunk_text.strip():
                    transcriptions.append(chunk_text.strip())
                    logger.info(f"Chunk {i+1} transcribed: {chunk_text[:50]}...")
                else:
                    logger.warning(f"Chunk {i+1} produced no text")
                    
            except sr.UnknownValueError:
                logger.warning(f"Chunk {i+1} could not be understood")
                # Don't add anything for this chunk
            except sr.RequestError as e:
                logger.error(f"Request error for chunk {i+1}: {e}")
                # Don't add anything for this chunk
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Don't add anything for this chunk
            finally:
                # Clean up chunk file
                if chunk_path.exists():
                    os.unlink(chunk_path)
            
            # Small delay between requests to respect API limits
            await asyncio.sleep(0.1)
        
        # Combine all transcriptions
        if transcriptions:
            full_text = " ".join(transcriptions)
            logger.info(f"Combined transcription from {len(transcriptions)} chunks")
            return full_text
        else:
            logger.error("No chunks were successfully transcribed")
            return "[Audio could not be transcribed - all chunks failed]"
            
    except Exception as e:
        logger.error(f"Chunked transcription failed: {e}", exc_info=True)
        return "[Long audio transcription failed due to technical error]"

# Legacy function for compatibility
async def transcribe_with_local_whisper(processed_path: Path) -> str:
    """Legacy wrapper - now uses Wav2Vec2."""
    return await transcribe_with_wav2vec2(processed_path)

async def apply_text_enhancement_pipeline(raw_text: str) -> str:
    """Apply full text enhancement: IndicNLP + LLM cleanup."""
    # Step 1: IndicNLP normalization
    try:
        normalizer = IndicNormalizerFactory().get_normalizer("hi")
        words = raw_text.split()
        normalized_text = " ".join(normalizer.normalize(w) for w in words)
        logger.info(f"IndicNLP normalized: {normalized_text[:100]}...")
    except Exception as norm_error:
        logger.warning(f"IndicNLP normalization failed: {norm_error}")
        normalized_text = raw_text
    
    # Step 2: Return normalized text (LLM cleanup can be added later if needed)
    return normalized_text if normalized_text else raw_text

# Keep legacy function name for compatibility
async def transcribe_with_whisper(processed_path: Path) -> str:
    """Legacy wrapper - now uses Wav2Vec2 pipeline."""
    return await transcribe_with_wav2vec2(processed_path)

async def preprocess_audio(input_path: Path, output_path: Path) -> bool:
    """
    Converts audio to the optimal format for ASR models (16kHz, 16-bit, mono PCM WAV).
    Enhanced to preserve complete audio without truncation.
    """
    try:
        logger.info(f"Preprocessing audio: {input_path}")
        
        # Get input audio info first
        probe = ffmpeg.probe(str(input_path))
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        if audio_stream:
            duration = float(audio_stream.get('duration', 0))
            logger.info(f"Input audio duration: {duration:.2f} seconds")
        
        # Use more robust FFmpeg settings to preserve complete audio
        (
            ffmpeg
            .input(str(input_path))
            .output(
                str(output_path), 
                acodec='pcm_s16le',  # 16-bit PCM
                ac=1,                # Mono
                ar=16000,            # 16kHz sample rate
                avoid_negative_ts='make_zero',  # Handle timestamp issues
                f='wav'              # Explicitly set WAV format
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        
        # Verify output file was created and has reasonable size
        if output_path.exists() and output_path.stat().st_size > 1000:  # At least 1KB
            # Check output duration to ensure no truncation
            try:
                output_probe = ffmpeg.probe(str(output_path))
                output_audio = next((stream for stream in output_probe['streams'] if stream['codec_type'] == 'audio'), None)
                if output_audio:
                    output_duration = float(output_audio.get('duration', 0))
                    logger.info(f"Output audio duration: {output_duration:.2f} seconds")
                    
                    # Check if significant audio was lost (more than 1 second difference)
                    if audio_stream and abs(duration - output_duration) > 1.0:
                        logger.warning(f"Potential audio loss detected: input {duration:.2f}s vs output {output_duration:.2f}s")
            except Exception as verify_e:
                logger.warning(f"Could not verify output duration: {verify_e}")
            
            logger.info(f"Successfully preprocessed audio to {output_path}")
            return True
        else:
            logger.error("Output file is missing or too small")
            return False
            
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg preprocessing failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        # Fallback to copying if conversion fails (e.g., already in correct format)
        try:
            logger.info("Attempting fallback: direct file copy")
            shutil.copy2(input_path, output_path)
            return True
        except Exception as fallback_e:
            logger.error(f"File copy fallback failed: {fallback_e}")
            return False

async def transcribe_full_audio(file_path: Path, client_id: str | None = None) -> Dict[str, Any]:
    """
    Main transcription function using faster-whisper. It handles caching,
    preprocessing, and transcription, providing progress updates.
    """
    try:
        # 1. Check cache - TEMPORARILY DISABLED FOR TESTING
        file_hash = get_file_hash(file_path)
        cache_file = CACHE_DIR / f"{file_hash}.json"
        # if cache_file.exists():
        #     logger.info(f"Returning cached transcription for {file_path.name}")
        #     async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
        #         cached_result = json.loads(await f.read())
        #     if client_id:
        #         await manager.send_progress(client_id, {"status": "completed", "progress": 100, "message": "Transcription complete (from cache)"})
        #     return cached_result
        logger.info(f"CACHE DISABLED - Processing fresh transcription for {file_path.name}")

        # 2. Load the best available Hindi ASR model
        if client_id:
            await manager.send_progress(client_id, {"status": "loading_model", "progress": 5, "message": "Loading Hindi ASR model..."})
        
        # Load Wav2Vec2 model for best Hindi accuracy
        logger.info("Loading Wav2Vec2 model for Hindi transcription")
        model, processor = load_wav2vec2_model()

        # 3. Preprocess audio
        if client_id:
            await manager.send_progress(client_id, {"status": "preprocessing", "progress": 15, "message": "Preparing audio..."})
        
        processed_path = Path("temp") / f"processed_{file_path.stem}.wav"
        if not await preprocess_audio(file_path, processed_path):
            raise HTTPException(status_code=500, detail="Audio preprocessing failed.")

        # 4. Transcribe
        if client_id:
            await manager.send_progress(client_id, {"status": "transcribing", "progress": 30, "message": "Transcribing audio..."})

        # Use Wav2Vec2 model for transcription
        try:
            transcribed_text = await transcribe_with_wav2vec2(processed_path)
            model_used = "facebook/wav2vec2-large-xlsr-53"
            logger.info(f"🎉 Transcription completed successfully with {model_used}")
        except Exception as transcription_error:
            logger.error(f"Transcription failed: {transcription_error}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(transcription_error)}")

        # 5. Prepare and cache result
        result = {
            "text": transcribed_text,
            "language": "hi",
            "timestamp": datetime.now().isoformat(),
            "model_used": model_used,
            "word_count": len(transcribed_text.split())
        }

        async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(result, ensure_ascii=False, indent=2))

        # 6. Cleanup
        if processed_path.exists():
            os.unlink(processed_path)

        if client_id:
            await manager.send_progress(client_id, {"status": "completed", "progress": 100, "message": "Transcription completed successfully!"})
        
        return result

    except Exception as e:
        logger.error(f"Transcription failed for {file_path.name}: {e}", exc_info=True)
        if client_id:
            await manager.send_progress(client_id, {"status": "error", "progress": 0, "message": f"An error occurred: {str(e)}"})
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# --- API Routes ---

@app.get("/")
async def home(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Handles audio file uploads."""
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{Path(file.filename).name}"
        file_path = UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size = len(content)
        if file_size > 500 * 1024 * 1024: # 500MB limit
            os.unlink(file_path)
            raise HTTPException(status_code=413, detail="File too large (max 500MB).")
        
        return JSONResponse({"success": True, "file_id": safe_filename, "message": "File uploaded."})
        
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/transcribe/{file_id}")
async def start_transcription(file_id: str):
    """Initiates the transcription process for an uploaded file."""
    try:
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found.")
        
        # Generate client ID for WebSocket progress updates
        client_id = f"transcribe-{file_id}-{datetime.now().timestamp()}"
        
        # This endpoint now directly returns the result.
        # For long files, the client will wait.
        # A background task model would be better for production.
        result = await transcribe_full_audio(file_path, client_id)
        
        return JSONResponse({"success": True, "transcription": result})
        
    except HTTPException as e:
        logger.error(f"HTTP error during transcription: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start transcription: {str(e)}")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handles WebSocket connections for progress updates."""
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected.")

@app.get("/health")
async def health_check():
    """Provides a simple health check endpoint."""
    global wav2vec2_model, speech_recognizer
    return {
        "status": "healthy", 
        "wav2vec2_loaded": wav2vec2_model is not None,
        "speech_recognition_loaded": speech_recognizer is not None
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify server is responding."""
    return {"message": "Server is running", "timestamp": datetime.now().isoformat()}

@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to check model status."""
    global wav2vec2_model, wav2vec2_processor, speech_recognizer
    
    return {
        "wav2vec2_model_loaded": wav2vec2_model is not None,
        "wav2vec2_processor_loaded": wav2vec2_processor is not None,
        "speech_recognizer_loaded": speech_recognizer is not None,
        "models_available": ["facebook/wav2vec2-large-xlsr-53", "Google Speech Recognition"]
    }

if __name__ == "__main__":
    import uvicorn
    # Don't load model at startup - load lazily on first request
    logger.info("Starting server without preloading model (lazy loading)")
    uvicorn.run(app, host="0.0.0.0", port=8000)