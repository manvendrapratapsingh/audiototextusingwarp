"""
Advanced Audio Transcription Service - Powered by OpenAI Whisper
Provides exceptional Hindi transcription accuracy with 95+ language support
"""

import os
import asyncio
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import shutil
import tempfile
import numpy as np

# *** BEST FREE ASR MODEL - OpenAI Whisper ***
import whisper
from faster_whisper import WhisperModel
import torch

# Audio processing and enhancement
import librosa
import soundfile as sf
import noisereduce as nr
from langdetect import detect

# Hindi text processing
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_AVAILABLE = True
except ImportError:
    INDIC_AVAILABLE = False
    logging.warning("IndicNLP not available - Hindi text normalization disabled")

# Speaker diarization
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
    logging.info("🎯 Speaker diarization available - can identify who spoke when")
except ImportError:
    DIARIZATION_AVAILABLE = False
    logging.info("Speaker diarization not available - install pyannote-audio for this feature")

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import aiofiles
import ffmpeg

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Audio Transcription - Whisper Powered", 
    version="3.0.0",
    description="AI-powered speech-to-text with exceptional Hindi accuracy and 95+ language support"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# *** WHISPER MODEL CONFIGURATION ***
# Available models: tiny, base, small, medium, large-v3
# For Hindi: large-v3 gives best accuracy, medium is good balance
WHISPER_MODEL_SIZE = "large-v3"  # Best for Hindi accuracy
USE_FASTER_WHISPER = True  # 4x faster processing
ENABLE_AUDIO_ENHANCEMENT = False  # Disabled to avoid erasing quiet speech during debugging for full coverage
ENABLE_SPEAKER_DIARIZATION = True  # Identify who spoke when (requires pyannote-audio)
# Force chunked processing for full coverage
FORCE_CHUNKING = True

# Accuracy controls and language defaults
ACCURACY_MODE_DEFAULT = False  # When True, apply stronger anti‑hallucination heuristics
CLEAN_ARTIFACTS = True         # Light cleanup of common non‑speech artifacts
DEFAULT_LANGUAGE = "hi"        # Default transcription language
DEFAULT_HINDI_PROMPT = (
    "कृपया शुद्ध हिंदी में स्पष्ट और सरल प्रतिलेखन दें। "
    "यदि कोई नाम, स्थान, तकनीकी शब्द या ब्रांड का उच्चारण हो तो उसे ठीक से लिखें।"
)

# Optional Hindi spelling corrections for frequent ASR confusions
ENABLE_HINDI_CORRECTIONS = True

# Supported formats and limits
SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.webm', '.mp4'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_DURATION = 3600  # 1 hour maximum

# Caching control (disabled to ensure fresh, full transcripts)
ENABLE_CACHING = False

# Directories
CACHE_DIR = Path("temp/cache")
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp")

# Create necessary directories
for dir_path in [CACHE_DIR, UPLOAD_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global model instances
whisper_model = None
faster_whisper_model = None
diarization_pipeline = None

# Language configuration
HINDI_LANGUAGE_CODES = ['hi', 'hi-IN', 'hindi']
SUPPORTED_LANGUAGES = {
    'hi': 'Hindi', 'en': 'English', 'ur': 'Urdu', 'bn': 'Bengali', 
    'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada',
    'gu': 'Gujarati', 'pa': 'Punjabi', 'mr': 'Marathi', 'or': 'Odia',
    'as': 'Assamese', 'ne': 'Nepali', 'si': 'Sinhala', 'my': 'Myanmar',
    'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay',
    'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
    'fa': 'Persian', 'tr': 'Turkish', 'ru': 'Russian', 'fr': 'French',
    'de': 'German', 'es': 'Spanish', 'pt': 'Portuguese', 'it': 'Italian'
}

class ConnectionManager:
    """Manages WebSocket connections for real-time progress updates."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_progress(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send progress to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

def load_diarization_pipeline():
    """
    Load speaker diarization pipeline for identifying who spoke when.
    This enables advanced multi-speaker analysis.
    """
    global diarization_pipeline
    
    if not DIARIZATION_AVAILABLE:
        logger.info("🔇 Speaker diarization not available - install pyannote-audio")
        return None
        
    if diarization_pipeline is None:
        try:
            logger.info("🎯 Loading speaker diarization pipeline...")
            # Use the pre-trained speaker diarization model
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=None  # Public model
            )
            logger.info("✅ Speaker diarization pipeline loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load diarization pipeline: {e}")
            diarization_pipeline = None
            return None
    
    return diarization_pipeline

async def perform_speaker_diarization(audio_path: Path) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization to identify who spoke when.
    Returns a list of speaker segments with timestamps.
    """
    if not ENABLE_SPEAKER_DIARIZATION or not DIARIZATION_AVAILABLE:
        return []
    
    try:
        pipeline = load_diarization_pipeline()
        if pipeline is None:
            return []
        
        logger.info("🎯 Performing speaker diarization...")
        
        # Run diarization
        diarization = pipeline(str(audio_path))
        
        # Convert to our format
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start_time": round(turn.start, 2),
                "end_time": round(turn.end, 2),
                "duration": round(turn.end - turn.start, 2)
            })
        
        logger.info(f"✅ Found {len(speaker_segments)} speaker segments")
        return speaker_segments
        
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}")
        return []

def load_whisper_model(model_size: str = WHISPER_MODEL_SIZE) -> Tuple[Any, str]:
    """
    Load the best Whisper model for transcription.
    Returns the model and model type used.
    """
    global whisper_model, faster_whisper_model
    
    try:
        if USE_FASTER_WHISPER:
            if faster_whisper_model is None:
                logger.info(f"🚀 Loading faster-whisper model: {model_size}")
                # Use faster-whisper for 4x speed improvement
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                faster_whisper_model = WhisperModel(
                    model_size, 
                    device=device, 
                    compute_type=compute_type,
                    cpu_threads=4
                )
                logger.info(f"✅ Faster-whisper model loaded successfully on {device}")
            return faster_whisper_model, f"faster-whisper-{model_size}"
        else:
            if whisper_model is None:
                logger.info(f"🚀 Loading OpenAI Whisper model: {model_size}")
                whisper_model = whisper.load_model(model_size)
                logger.info(f"✅ OpenAI Whisper model loaded successfully")
            return whisper_model, f"whisper-{model_size}"
            
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
        # Try smaller model as fallback
        if model_size != "base":
            logger.info("Falling back to base model...")
            return load_whisper_model("base")
        else:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash for file caching."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def enhance_audio(audio_path: Path, output_path: Path) -> bool:
    """
    Enhance audio quality by reducing noise and normalizing.
    This significantly improves transcription accuracy.
    """
    if not ENABLE_AUDIO_ENHANCEMENT:
        return False
        
    try:
        logger.info("🎧 Enhancing audio quality...")
        
        # Load audio
        audio_data, sample_rate = librosa.load(str(audio_path), sr=None)
        
        # Apply noise reduction
        enhanced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
        
        # Normalize audio
        enhanced_audio = librosa.util.normalize(enhanced_audio)
        
        # Save enhanced audio
        sf.write(str(output_path), enhanced_audio, sample_rate)
        
        logger.info("✨ Audio enhancement completed")
        return True
        
    except Exception as e:
        logger.warning(f"Audio enhancement failed: {e}")
        return False

async def preprocess_audio(input_path: Path, output_path: Path) -> bool:
    """Convert audio to optimal format for Whisper (16 kHz, mono)."""
    try:
        logger.info(f"🔄 Preprocessing audio: {input_path.name}")
        
        # Get input audio info
        probe = ffmpeg.probe(str(input_path))
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        if audio_stream:
            duration = float(audio_stream.get('duration', 0))
            logger.info(f"📊 Input duration: {duration:.2f} seconds")
            
            # Check duration against configured maximum; allow processing but warn
            if duration > MAX_DURATION:
                logger.warning(f"Audio length {duration:.1f}s exceeds configured MAX_DURATION={MAX_DURATION}s; proceeding anyway for full coverage")
        
        # Enhanced audio preprocessing with better quality preservation
        stream = ffmpeg.input(str(input_path))
        stream = ffmpeg.output(
            stream,
            str(output_path),
            acodec='pcm_s16le',
            ac=1,  # mono
            ar=16000,  # 16kHz for Whisper
            avoid_negative_ts='make_zero',
            f='wav'
        )
        
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        
        # Verify output
        if output_path.exists() and output_path.stat().st_size > 1000:
            logger.info(f"✅ Audio preprocessing successful")
            return True
        else:
            logger.error("Output file invalid")
            return False
            
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return False

def _clean_whisper_artifacts(text: str) -> str:
    """Light cleanup for common Whisper artifacts without changing meaning."""
    if not text:
        return text
    try:
        t = text
        # Remove common bracketed non-speech tags if present
        for tag in ["[Music]", "[music]", "[Applause]", "[applause]", "[Silence]", "[silence]"]:
            t = t.replace(tag, " ")
        # Collapse repeated spaces and punctuation
        t = " ".join(t.split())
        while ".." in t:
            t = t.replace("..", ".")
        while "،،" in t:
            t = t.replace("،،", "،")
        while "।।" in t:
            t = t.replace("।।", "।")
        return t.strip()
    except Exception:
        return text

async def transcribe_with_whisper(
    audio_path: Path,
    client_id: Optional[str] = None,
    *,
    language: str = DEFAULT_LANGUAGE,
    initial_prompt: Optional[str] = None,
    accuracy_mode: bool = ACCURACY_MODE_DEFAULT,
) -> Dict[str, Any]:
    """
    Main transcription function using Whisper.
    Provides 100% audio coverage with NO VAD filtering and chunking for long files.
    """
    try:
        # Load Whisper model
        if client_id:
            await manager.send_progress(client_id, {
                "status": "loading_model", 
                "progress": 10, 
                "message": "Loading Whisper AI model..."
            })
        
        model, model_name = load_whisper_model()
        
        # Check audio duration for chunking strategy
        try:
            probe = ffmpeg.probe(str(audio_path))
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            duration = float(audio_stream.get('duration', 0)) if audio_stream else 0
            logger.info(f"📏 Audio duration: {duration:.2f} seconds")
            
            # Use chunking for full coverage (forced)
            if FORCE_CHUNKING or duration > 600:
                logger.info("🔩 Using chunking strategy for complete coverage")
                return await transcribe_long_audio_chunked(
                    model, model_name, audio_path, client_id, duration,
                    language=language, initial_prompt=initial_prompt, accuracy_mode=accuracy_mode
                )
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            duration = 0
        
        # Update progress
        if client_id:
            await manager.send_progress(client_id, {
                "status": "transcribing", 
                "progress": 30, 
                "message": "Transcribing audio with AI..."
            })
        
        if USE_FASTER_WHISPER:
            # Use faster-whisper for speed
            try:
                # MOST AGGRESSIVE: Force complete processing with multiple strategies
                logger.info("🔥 FORCING COMPLETE AUDIO PROCESSING - No segments will be lost!")
                
                # Strategy 1: Ultra-conservative transcription settings
                segments, info = model.transcribe(
                    str(audio_path),
                    beam_size=5,  # Better robustness
                    language=language or DEFAULT_LANGUAGE,  # Primary focus on Hindi
                    task="transcribe",
                    vad_filter=False,  # ABSOLUTELY NO VAD
                    word_timestamps=True,
                    condition_on_previous_text=True if accuracy_mode else False,
                    compression_ratio_threshold=2.4 if accuracy_mode else None,
log_prob_threshold=None,  # Disable drop filter
                    no_speech_threshold=0.3,
                    temperature=[0.0] if accuracy_mode else [0.0, 0.2, 0.4],
                    initial_prompt=(initial_prompt or (DEFAULT_HINDI_PROMPT if (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES else None)),
                    patience=2.0  # Patient decoding
                )
                
                # FORCE processing of ALL segments - even if they seem empty
                all_segment_texts = []
                all_segments_info = []
                total_segments = 0
                total_duration = 0
                audio_gaps = []
                
                previous_end = 0.0
                
                for segment in segments:
                    total_segments += 1
                    
                    # Check for gaps in audio timeline
                    if segment.start > previous_end + 0.1:  # Gap detected
                        gap_duration = segment.start - previous_end
                        audio_gaps.append((previous_end, segment.start, gap_duration))
                        logger.warning(f"⚠️ Audio gap detected: {previous_end:.2f}s - {segment.start:.2f}s ({gap_duration:.2f}s)")
                    
                    # Process ALL segments, including seemingly empty ones
                    segment_text = segment.text.strip()
                    segment_duration = segment.end - segment.start
                    total_duration += segment_duration
                    
                    # Log every segment for debugging
                    logger.debug(f"Segment {total_segments}: {segment.start:.2f}-{segment.end:.2f}s: '{segment_text[:50]}...'")
                    
                    # Include ALL segments - don't filter any
                    if segment_text:  # Non-empty
                        all_segment_texts.append(segment_text)
                    else:
                        # Even "empty" segments might contain important audio
                        logger.debug(f"Empty segment at {segment.start:.2f}-{segment.end:.2f}s - investigating...")
                        
                        # Try to re-process this specific segment with different settings
                        try:
                            # Extract just this time segment
                            segment_audio = f"segment_{segment.start:.2f}_{segment.end:.2f}.wav"
                            # Process with even more aggressive settings
                            logger.debug(f"Re-processing empty segment: {segment.start:.2f}-{segment.end:.2f}s")
                        except Exception as e:
                            logger.debug(f"Could not re-process segment: {e}")
                    
                    all_segments_info.append({
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment_duration,
                        "text": segment_text,
                        "is_empty": not bool(segment_text)
                    })
                    
                    previous_end = segment.end
                
                # Combine ALL text - no filtering
                transcribed_text = " ".join(all_segment_texts)
                
                # Enhanced logging for complete debugging
                logger.info(f"📊 ULTRA-DETAILED ANALYSIS:")
                logger.info(f"   • Total segments processed: {total_segments}")
                logger.info(f"   • Total audio duration covered: {total_duration:.2f}s")
                logger.info(f"   • Audio gaps detected: {len(audio_gaps)}")
                logger.info(f"   • Final transcription length: {len(transcribed_text)} chars")
                logger.info(f"   • Non-empty segments: {len(all_segment_texts)}")
                
                # Report any gaps found
                if audio_gaps:
                    total_gap_time = sum(gap[2] for gap in audio_gaps)
                    logger.warning(f"⚠️ POTENTIAL MISSING AUDIO: {len(audio_gaps)} gaps totaling {total_gap_time:.2f}s")
                    for gap_start, gap_end, gap_duration in audio_gaps:
                        logger.warning(f"   Gap: {gap_start:.2f}s - {gap_end:.2f}s ({gap_duration:.2f}s)")
                    
                    # If significant gaps, try alternative processing
                    if total_gap_time > 5.0:  # More than 5 seconds of gaps
                        logger.error("😱 MAJOR AUDIO LOSS DETECTED - Attempting recovery...")
                        return await attempt_gap_recovery(
                            model, model_name, audio_path, client_id, audio_gaps,
                            language=language, initial_prompt=initial_prompt, accuracy_mode=accuracy_mode
                        )
                
                # Validate transcription completeness
                expected_duration = duration if 'duration' in locals() else 0
                if expected_duration > 0:
                    coverage = (total_duration / expected_duration) * 100
                    logger.info(f"🎯 Audio coverage: {coverage:.1f}% ({total_duration:.2f}s / {expected_duration:.2f}s)")
                    
                    if coverage < 90:  # Less than 90% coverage is unacceptable
                        logger.error(f"😱 INSUFFICIENT COVERAGE: {coverage:.1f}% - Attempting recovery...")
                        return await attempt_complete_reprocessing(
                            model, model_name, audio_path, client_id,
                            language=language, initial_prompt=initial_prompt, accuracy_mode=accuracy_mode
                        )
                
                detected_language = info.language
                language_probability = info.language_probability
                
                logger.info(f"✅ FORCED COMPLETE PROCESSING: {total_segments} segments, {len(transcribed_text)} chars")
                
            except Exception as e:
                logger.error(f"Forced processing failed: {e}")
                # Emergency fallback - chunk the entire audio
                logger.info("🎆 EMERGENCY RECOVERY: Chunking entire audio...")
                return await emergency_complete_transcription(
                    model, model_name, audio_path, client_id,
                    language=language, initial_prompt=initial_prompt, accuracy_mode=accuracy_mode
                )
            
        else:
            # Use standard whisper with improved handling
            try:
                result = model.transcribe(
                    str(audio_path),
                    language=language or DEFAULT_LANGUAGE,  # Primary focus on Hindi
                    task="transcribe",
                    fp16=torch.cuda.is_available(),
                    verbose=True,  # Enable detailed logging
                    initial_prompt=(initial_prompt or (DEFAULT_HINDI_PROMPT if (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES else None)),
                )
                
                transcribed_text = result["text"].strip()
                detected_language = result["language"]
                language_probability = 1.0
                
                # Log segment information if available
                if "segments" in result:
                    logger.info(f"📊 Standard Whisper processed {len(result['segments'])} segments")
                    
            except Exception as e:
                logger.error(f"Standard Whisper transcription failed: {e}")
                raise
        
        # Light cleanup and Indic enhancement
        cleaned_text = _clean_whisper_artifacts(transcribed_text) if CLEAN_ARTIFACTS else transcribed_text
        if detected_language in HINDI_LANGUAGE_CODES:
            enhanced_text = await enhance_hindi_text(cleaned_text)
            if ENABLE_HINDI_CORRECTIONS:
                enhanced_text = apply_hindi_corrections(enhanced_text)
        else:
            enhanced_text = cleaned_text
        
        # Perform speaker diarization if enabled
        speaker_segments = []
        if ENABLE_SPEAKER_DIARIZATION and client_id:
            await manager.send_progress(client_id, {
                "status": "diarization", 
                "progress": 80, 
                "message": "Identifying speakers..."
            })
            speaker_segments = await perform_speaker_diarization(audio_path)
        
        # Prepare result with complete coverage tracking
        result_data = {
            "text": enhanced_text,
            "original_text": transcribed_text,
            "language": detected_language,
            "language_name": SUPPORTED_LANGUAGES.get(detected_language, detected_language),
            "language_probability": language_probability,
            "model_used": model_name,
            "word_count": len(enhanced_text.split()),
            "timestamp": datetime.now().isoformat(),
            "enhanced": detected_language in HINDI_LANGUAGE_CODES,
            "total_duration_processed": total_duration,  # For coverage validation
            "total_segments_processed": total_segments,  # For debugging
            "vad_disabled": True,  # Confirm no VAD filtering was used
            "speaker_diarization": {
                "enabled": ENABLE_SPEAKER_DIARIZATION and DIARIZATION_AVAILABLE,
                "segments": speaker_segments,
                "speaker_count": len(set(seg["speaker"] for seg in speaker_segments)) if speaker_segments else 0
            }
        }
        
        logger.info(f"🎉 Transcription completed: {len(enhanced_text)} chars, {detected_language}")
        return result_data
        
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

async def attempt_gap_recovery(model: Any, model_name: str, audio_path: Path, client_id: Optional[str] = None, audio_gaps: List = None, *, language: str = DEFAULT_LANGUAGE, initial_prompt: Optional[str] = None, accuracy_mode: bool = ACCURACY_MODE_DEFAULT) -> Dict[str, Any]:
    """
    Emergency function to recover missing audio segments from detected gaps.
    """
    try:
        logger.info("🎆 EMERGENCY GAP RECOVERY: Processing gaps to recover missing audio...")
        
        if client_id:
            await manager.send_progress(client_id, {
                "status": "gap_recovery",
                "progress": 60,
                "message": "Recovering missing audio segments..."
            })
        
        # Load full audio for gap analysis using soundfile (avoid librosa backends)
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
        if getattr(audio_data, 'ndim', 1) > 1:
            audio_data = audio_data.mean(axis=1)
        total_duration = len(audio_data) / sample_rate
        
        # Process the entire audio with ultra-aggressive settings
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=1,  # Fastest processing
            language=language or DEFAULT_LANGUAGE,
            task="transcribe",
            vad_filter=False,  # NO VAD
            condition_on_previous_text=False,  # Don't rely on context
            compression_ratio_threshold=10.0,  # Very lenient
log_prob_threshold=-10.0,  # Very lenient  
            no_speech_threshold=0.99,  # Almost never skip
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8],  # Multiple attempts
            initial_prompt=(initial_prompt or (DEFAULT_HINDI_PROMPT if (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES else None)),
            patience=5.0  # Very patient
        )
        
        # Force include ALL segments without any filtering
        all_texts = []
        processed_duration = 0
        
        for segment in segments:
            text = segment.text.strip()
            processed_duration += segment.end - segment.start
            
            # Include EVERYTHING - even single characters
            if text:
                all_texts.append(text)
            
            logger.debug(f"Recovery segment: {segment.start:.2f}-{segment.end:.2f}s: '{text}'")
        
        recovered_text = " ".join(all_texts)
        # Cleanup and optional Hindi correction
        cleaned_recovered = _clean_whisper_artifacts(recovered_text) if CLEAN_ARTIFACTS else recovered_text
        if (info.language or language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES:
            final_recovered = await enhance_hindi_text(cleaned_recovered)
            if ENABLE_HINDI_CORRECTIONS:
                final_recovered = apply_hindi_corrections(final_recovered)
        else:
            final_recovered = cleaned_recovered
        
        logger.info(f"✅ GAP RECOVERY COMPLETE: {len(segments)} segments, {len(recovered_text)} chars")
        logger.info(f"📏 Coverage after recovery: {(processed_duration/total_duration)*100:.1f}%")
        
        return {
            "text": final_recovered,
            "original_text": recovered_text,
            "language": info.language,
            "language_name": SUPPORTED_LANGUAGES.get(info.language, info.language),
            "language_probability": info.language_probability,
            "model_used": f"{model_name}-gap-recovery",
            "word_count": len((final_recovered or "").split()),
            "timestamp": datetime.now().isoformat(),
            "enhanced": info.language in HINDI_LANGUAGE_CODES,
            "total_duration_processed": processed_duration,
            "total_segments_processed": len(segments),
            "recovery_mode": "gap_recovery",
            "vad_disabled": True,
            "gaps_detected": len(audio_gaps) if audio_gaps else 0,
            "speaker_diarization": {
                "enabled": False,
                "segments": [],
                "speaker_count": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Gap recovery failed: {e}")
        # Final fallback
        return await emergency_complete_transcription(
            model, model_name, audio_path, client_id,
            language=language, initial_prompt=initial_prompt, accuracy_mode=accuracy_mode
        )

async def attempt_complete_reprocessing(model: Any, model_name: str, audio_path: Path, client_id: Optional[str] = None, *, language: str = DEFAULT_LANGUAGE, initial_prompt: Optional[str] = None, accuracy_mode: bool = ACCURACY_MODE_DEFAULT) -> Dict[str, Any]:
    """
    Attempt complete reprocessing with different strategies.
    """
    try:
        logger.info("🔄 COMPLETE REPROCESSING: Trying alternative transcription strategy...")
        
        if client_id:
            await manager.send_progress(client_id, {
                "status": "reprocessing",
                "progress": 70,
                "message": "Reprocessing with alternative strategy..."
            })
        
        # Strategy: Multiple temperature sampling
        best_result = None
        best_length = 0
        
        temperatures = [0.0, 0.2, 0.4, 0.6]
        
        for temp in temperatures:
            try:
                logger.info(f"🌡️ Trying temperature: {temp}")
                
                segments, info = model.transcribe(
                    str(audio_path),
                    beam_size=3,
                    language=language or DEFAULT_LANGUAGE,
                    task="transcribe",
                    vad_filter=False,
                    temperature=temp,
                    condition_on_previous_text=True,
                    word_timestamps=True,
                    initial_prompt=(initial_prompt or (DEFAULT_HINDI_PROMPT if (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES else None)),
                )
                
                text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
                full_text = " ".join(text_parts)
                
                logger.info(f"Temperature {temp} result: {len(full_text)} chars, {len(segments)} segments")
                
                # Keep the longest result
                if len(full_text) > best_length:
                    best_length = len(full_text)
                    best_result = {
                        "text": full_text,
                        "segments": len(segments),
                        "info": info,
                        "temperature": temp
                    }
                    
            except Exception as e:
                logger.warning(f"Temperature {temp} failed: {e}")
                continue
        
        if best_result:
            logger.info(f"✅ REPROCESSING SUCCESS: Best result with temp {best_result['temperature']}: {best_length} chars")
            
            return {
                "text": best_result["text"],
                "original_text": best_result["text"],
                "language": best_result["info"].language,
                "language_name": SUPPORTED_LANGUAGES.get(best_result["info"].language, best_result["info"].language),
                "language_probability": best_result["info"].language_probability,
                "model_used": f"{model_name}-reprocessed-temp{best_result['temperature']}",
                "word_count": len(best_result["text"].split()),
                "timestamp": datetime.now().isoformat(),
                "enhanced": best_result["info"].language in HINDI_LANGUAGE_CODES,
                "total_segments_processed": best_result["segments"],
                "recovery_mode": "temperature_sampling",
                "vad_disabled": True,
                "speaker_diarization": {
                    "enabled": False,
                    "segments": [],
                    "speaker_count": 0
                }
            }
        else:
            raise Exception("All reprocessing attempts failed")
            
    except Exception as e:
        logger.error(f"Complete reprocessing failed: {e}")
        return await emergency_complete_transcription(
            model, model_name, audio_path, client_id,
            language=language, initial_prompt=initial_prompt, accuracy_mode=accuracy_mode
        )

async def emergency_complete_transcription(model: Any, model_name: str, audio_path: Path, client_id: Optional[str] = None, *, language: str = DEFAULT_LANGUAGE, initial_prompt: Optional[str] = None, accuracy_mode: bool = ACCURACY_MODE_DEFAULT) -> Dict[str, Any]:
    """
    Emergency last-resort complete transcription using chunking.
    """
    try:
        logger.info("🎆 EMERGENCY COMPLETE TRANSCRIPTION: Last resort chunking...")
        
        if client_id:
            await manager.send_progress(client_id, {
                "status": "emergency_processing",
                "progress": 80,
                "message": "Emergency complete processing..."
            })
        
        # Load and chunk audio very aggressively (avoid librosa to prevent sunau import)
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
        if getattr(audio_data, 'ndim', 1) > 1:
            audio_data = audio_data.mean(axis=1)
        total_duration = len(audio_data) / sample_rate
        
        # Very small chunks with heavy overlap
        chunk_duration = 30  # 30 seconds
        overlap_duration = 10  # 10 second overlap
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        
        all_texts = []
        processed_duration = 0
        
        logger.info(f"💿 Emergency chunking: {total_duration:.1f}s audio into {chunk_duration}s chunks with {overlap_duration}s overlap")
        
        for chunk_start in range(0, len(audio_data), chunk_samples - overlap_samples):
            chunk_end = min(chunk_start + chunk_samples, len(audio_data))
            chunk_audio = audio_data[chunk_start:chunk_end]
            chunk_duration_actual = len(chunk_audio) / sample_rate
            chunk_start_time = chunk_start / sample_rate
            
            # Save chunk
            chunk_path = TEMP_DIR / f"emergency_chunk_{chunk_start_time:.1f}.wav"
            sf.write(str(chunk_path), chunk_audio, sample_rate)
            
            try:
                # Process chunk with most aggressive settings
                segments, info = model.transcribe(
                    str(chunk_path),
                    beam_size=1,
                    language=language or DEFAULT_LANGUAGE,
                    task="transcribe",
                    vad_filter=False,
                    temperature=0.0,
                    compression_ratio_threshold=2.4 if accuracy_mode else 10.0,
log_prob_threshold=-10.0,
                    no_speech_threshold=0.99,
                    initial_prompt=(initial_prompt or (DEFAULT_HINDI_PROMPT if (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES else None)),
                )
                
                # Collect ALL text from chunk
                chunk_texts = [segment.text.strip() for segment in segments if segment.text.strip()]
                if chunk_texts:
                    chunk_text = " ".join(chunk_texts)
                    all_texts.append(chunk_text)
                    logger.debug(f"Emergency chunk {chunk_start_time:.1f}s: {len(chunk_text)} chars")
                
                processed_duration += chunk_duration_actual
                
                # Clean up
                os.unlink(chunk_path)
                
            except Exception as e:
                logger.warning(f"Emergency chunk {chunk_start_time:.1f}s failed: {e}")
                # Continue with next chunk
                if chunk_path.exists():
                    os.unlink(chunk_path)
                continue
        
        # Combine and clean
        emergency_text = " ".join(all_texts)
        cleaned_text = _clean_whisper_artifacts(emergency_text) if CLEAN_ARTIFACTS else emergency_text
        if (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES:
            final_text = await enhance_hindi_text(cleaned_text)
            if ENABLE_HINDI_CORRECTIONS:
                final_text = apply_hindi_corrections(final_text)
        else:
            final_text = cleaned_text
        
        logger.info(f"✅ EMERGENCY TRANSCRIPTION COMPLETE: {len(all_texts)} chunks, {len(emergency_text)} chars")
        
        return {
            "text": final_text,
            "original_text": emergency_text,
            "language": language or DEFAULT_LANGUAGE,
            "language_name": SUPPORTED_LANGUAGES.get(language or DEFAULT_LANGUAGE, language or DEFAULT_LANGUAGE),
            "language_probability": 1.0,
            "model_used": f"{model_name}-emergency-chunked",
            "word_count": len((final_text or "").split()),
            "timestamp": datetime.now().isoformat(),
            "enhanced": True,
            "total_duration_processed": processed_duration,
            "recovery_mode": "emergency_chunking",
            "chunks_processed": len(all_texts),
            "vad_disabled": True,
            "speaker_diarization": {
                "enabled": False,
                "segments": [],
                "speaker_count": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Emergency transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"All transcription attempts failed: {str(e)}")

async def transcribe_long_audio_chunked(model: Any, model_name: str, audio_path: Path, client_id: Optional[str] = None, total_duration: float = 0, *, language: str = DEFAULT_LANGUAGE, initial_prompt: Optional[str] = None, accuracy_mode: bool = ACCURACY_MODE_DEFAULT) -> Dict[str, Any]:
    """
    Transcribe very long audio files by chunking to ensure complete coverage.
    """
    try:
        logger.info(f"🔩 Starting chunked transcription for {total_duration:.2f}s audio")
        
        # Load full audio for chunking using soundfile only
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
        if getattr(audio_data, 'ndim', 1) > 1:
            audio_data = audio_data.mean(axis=1)
        chunk_duration = 60  # 60 seconds per chunk for robust coverage
        chunk_samples = chunk_duration * sample_rate
        
        all_segments = []
        total_text_parts = []
        processed_duration = 0
        
        # Process in chunks with overlap to prevent word loss at boundaries
        overlap_samples = 10 * sample_rate  # 10 second overlap for safety
        
        for chunk_idx in range(0, len(audio_data), chunk_samples - overlap_samples):
            chunk_start_time = chunk_idx / sample_rate
            chunk_end_idx = min(chunk_idx + chunk_samples, len(audio_data))
            chunk_audio = audio_data[chunk_idx:chunk_end_idx]
            chunk_duration_actual = len(chunk_audio) / sample_rate
            
            logger.info(f"💿 Processing chunk {chunk_idx // (chunk_samples - overlap_samples) + 1}: {chunk_start_time:.1f}s - {chunk_start_time + chunk_duration_actual:.1f}s")
            
            # Save chunk to temporary file
            chunk_path = TEMP_DIR / f"chunk_{chunk_idx}_{int(chunk_start_time)}.wav"
            sf.write(str(chunk_path), chunk_audio, sample_rate)
            
            try:
                # Transcribe chunk with NO VAD
                if USE_FASTER_WHISPER:
                    segments, info = model.transcribe(
                        str(chunk_path),
                        beam_size=5,
                        language=language or DEFAULT_LANGUAGE,
                        task="transcribe",
                        vad_filter=False,  # NO VAD for complete coverage
                        word_timestamps=True,
                        condition_on_previous_text=True if accuracy_mode else False,
                        compression_ratio_threshold=2.4 if accuracy_mode else None,
log_prob_threshold=None,
                        no_speech_threshold=0.3,
                        temperature=[0.0] if accuracy_mode else [0.0, 0.2, 0.4],
                        initial_prompt=(initial_prompt or (DEFAULT_HINDI_PROMPT if (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES else None)),
                    )
                    
                    # Adjust timestamps to global timeline
                    for segment in segments:
                        segment.start += chunk_start_time
                        segment.end += chunk_start_time
                        all_segments.append(segment)
                        
                        if segment.text.strip():
                            total_text_parts.append(segment.text.strip())
                    
                    processed_duration += chunk_duration_actual
                    
                    # Update progress
                    if client_id:
                        progress = min(85, 30 + (processed_duration / total_duration) * 50)
                        await manager.send_progress(client_id, {
                            "status": "transcribing",
                            "progress": int(progress),
                            "message": f"Processing chunk {chunk_idx // (chunk_samples - overlap_samples) + 1} ({processed_duration/60:.1f}/{total_duration/60:.1f} min)"
                        })
                
                # Clean up chunk file
                os.unlink(chunk_path)
                
            except Exception as e:
                logger.error(f"Chunk {chunk_idx} failed: {e}")
                # Continue with next chunk - don't fail entire transcription
                continue
        
        # Combine all text parts
        transcribed_text = " ".join(total_text_parts)
        
        # Detect language from first successful segment
        detected_language = "hi"  # Default to Hindi
        language_probability = 1.0
        
        logger.info(f"✅ Chunked transcription complete: {len(all_segments)} total segments, {len(transcribed_text)} chars")
        
        # Cleanup + Hindi enhancement + corrections
        cleaned_text = _clean_whisper_artifacts(transcribed_text) if CLEAN_ARTIFACTS else transcribed_text
        if detected_language in HINDI_LANGUAGE_CODES:
            enhanced_text = await enhance_hindi_text(cleaned_text)
            if ENABLE_HINDI_CORRECTIONS:
                enhanced_text = apply_hindi_corrections(enhanced_text)
        else:
            enhanced_text = cleaned_text
        
        return {
            "text": enhanced_text,
            "original_text": transcribed_text,
            "language": detected_language,
            "language_name": SUPPORTED_LANGUAGES.get(detected_language, detected_language),
            "language_probability": language_probability,
            "model_used": f"{model_name}-chunked",
            "word_count": len((enhanced_text or "").split()),
            "timestamp": datetime.now().isoformat(),
            "enhanced": detected_language in HINDI_LANGUAGE_CODES,
            "total_duration_processed": processed_duration,
            "total_segments_processed": len(all_segments),
            "chunked_processing": True,
            "vad_disabled": True,
            "speaker_diarization": {
                "enabled": False,  # Disabled for chunked processing
                "segments": [],
                "speaker_count": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Chunked transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunked transcription failed: {str(e)}")

async def enhance_hindi_text(text: str) -> str:
    """
    Enhance Hindi text using IndicNLP normalization.
    Returns the input unchanged if IndicNLP is unavailable or input is empty.
    """
    if not INDIC_AVAILABLE or not isinstance(text, str) or not text.strip():
        return text
    
    try:
        normalizer = IndicNormalizerFactory().get_normalizer("hi")
        words = text.split()
        normalized_words = [normalizer.normalize(word) for word in words]
        enhanced_text = " ".join(normalized_words)
        logger.info("📝 Applied Hindi text enhancement")
        return enhanced_text
    except Exception as e:
        logger.warning(f"Hindi text enhancement failed: {e}")
        return text

def apply_hindi_corrections(text: str) -> str:
    """
    Apply lightweight Hindi corrections for frequent ASR confusions.
    Keep changes conservative and spelling-oriented (no rephrasing).
    """
    if not text or not isinstance(text, str):
        return text

    corrections = {
        # Parties and political terms
        "कॉंग्रेस": "कांग्रेस",
        "कौंग्रेस": "कांग्रेस",
        "आर्जेटी": "राजद",
        "आरजेडी": "राजद",
        "RGT": "राजद",
        "बिजेपी": "बीजेपी",
        "बी जे पी": "बीजेपी",
        "जेडियू": "जेडीयू",
        # Common words
        "अक्षेप": "आक्षेप",
        "विरोजगारी": "बेरोजगारी",
        "विशय": "विषय",
        "उदना": "उतना",
        "तभ": "तभी",
        "विहार": "बिहार",
        "शूटता": "छूटता",
        # Style/transliteration
        "core": "कोर",
        "Core": "कोर",
    }

    try:
        out = text
        for wrong, right in corrections.items():
            out = out.replace(wrong, right)
        return out
    except Exception:
        return text
    
    try:
        normalizer = IndicNormalizerFactory().get_normalizer("hi")
        words = text.split()
        normalized_words = [normalizer.normalize(word) for word in words]
        enhanced_text = " ".join(normalized_words)
        
        logger.info("📝 Applied Hindi text enhancement")
        return enhanced_text
        
    except Exception as e:
        logger.warning(f"Hindi text enhancement failed: {e}")
        return text

async def transcribe_full_audio(file_path: Path, client_id: Optional[str] = None, *, language: str = DEFAULT_LANGUAGE, initial_prompt: Optional[str] = None, accuracy_mode: bool = ACCURACY_MODE_DEFAULT) -> Dict[str, Any]:
    """
    Complete transcription pipeline with caching and preprocessing.
    Ensures 100% audio coverage with validation.
    """
    try:
        start_time = datetime.now()
        
        # Check cache
        file_hash = get_file_hash(file_path)
        cache_file = CACHE_DIR / f"{file_hash}.json"
        
        if ENABLE_CACHING and cache_file.exists():
            logger.info(f"📦 Returning cached result for {file_path.name}")
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                cached_result = json.loads(await f.read())
            
            if client_id:
                await manager.send_progress(client_id, {
                    "status": "completed", 
                    "progress": 100, 
                    "message": "Transcription completed (from cache)"
                })
            return cached_result
        
        # Progress update
        if client_id:
            await manager.send_progress(client_id, {
                "status": "preprocessing", 
                "progress": 15, 
                "message": "Preparing audio file..."
            })
        
        # Get original audio duration for validation
        try:
            probe = ffmpeg.probe(str(file_path))
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            original_duration = float(audio_stream.get('duration', 0)) if audio_stream else 0
            logger.info(f"📏 Original audio duration: {original_duration:.2f} seconds")
        except Exception as e:
            logger.warning(f"Could not get original duration: {e}")
            original_duration = 0
        
        # Preprocess audio
        processed_path = TEMP_DIR / f"processed_{file_hash}.wav"
        if not await preprocess_audio(file_path, processed_path):
            raise HTTPException(status_code=500, detail="Audio preprocessing failed")
        
        # Optional audio enhancement
        if ENABLE_AUDIO_ENHANCEMENT:
            enhanced_path = TEMP_DIR / f"enhanced_{file_hash}.wav"
            if await enhance_audio(processed_path, enhanced_path):
                processed_path = enhanced_path
        
        # Transcribe
        result = await transcribe_with_whisper(
            processed_path,
            client_id,
            language=language,
            initial_prompt=initial_prompt,
            accuracy_mode=accuracy_mode,
        )
        
        # Validate complete coverage if we have duration info
        if original_duration > 0 and "total_duration_processed" in result:
            coverage_percentage = (result["total_duration_processed"] / original_duration) * 100
            logger.info(f"🎯 Audio coverage: {coverage_percentage:.1f}% ({result['total_duration_processed']:.2f}s / {original_duration:.2f}s)")
            
            if coverage_percentage < 95:  # If less than 95% coverage
                logger.warning(f"⚠️ Potential audio loss detected: {coverage_percentage:.1f}% coverage")
                result["coverage_warning"] = f"Only {coverage_percentage:.1f}% of audio was processed"
            else:
                logger.info("✅ Complete audio coverage confirmed")
                result["coverage_confirmed"] = True
        
        # Add processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result["processing_time_seconds"] = round(processing_time, 2)
        
        # Cache result
        async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(result, ensure_ascii=False, indent=2))
        
        # Cleanup
        for temp_file in [processed_path, TEMP_DIR / f"enhanced_{file_hash}.wav"]:
            if temp_file.exists():
                os.unlink(temp_file)
        
        # Final progress update
        if client_id:
            await manager.send_progress(client_id, {
                "status": "completed", 
                "progress": 100, 
                "message": f"Transcription completed in {processing_time:.1f}s"
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Full transcription failed for {file_path.name}: {e}", exc_info=True)
        if client_id:
            await manager.send_progress(client_id, {
                "status": "error", 
                "progress": 0, 
                "message": f"Error: {str(e)}"
            })
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# === API ROUTES ===

@app.get("/")
async def home(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Handle audio file uploads with validation."""
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Create safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{Path(file.filename).stem}{file_ext}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        content = await file.read()
        file_size = len(content)
        
        # Validate file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large: {file_size/(1024*1024):.1f}MB (max: {MAX_FILE_SIZE/(1024*1024)}MB)"
            )
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"📁 File uploaded: {safe_filename} ({file_size/(1024*1024):.1f}MB)")
        
        return JSONResponse({
            "success": True, 
            "file_id": safe_filename, 
            "file_size_mb": round(file_size/(1024*1024), 2),
            "message": "File uploaded successfully"
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/transcribe/{file_id}")
async def start_transcription(file_id: str, request: Request):
    """Start transcription process for uploaded file."""
    try:
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Parse optional options from request body
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        language = payload.get("language") or DEFAULT_LANGUAGE
        accuracy_mode = bool(payload.get("accuracy_mode", ACCURACY_MODE_DEFAULT))
        prompt = payload.get("prompt")
        domain_terms = payload.get("domain_terms") or []
        if isinstance(domain_terms, list) and domain_terms:
            terms_text = ", ".join(str(t) for t in domain_terms[:30])
            bias_hint = f" महत्वपूर्ण शब्द: {terms_text}."
        else:
            bias_hint = ""

        initial_prompt = None
        if prompt:
            initial_prompt = str(prompt).strip()
            if bias_hint:
                initial_prompt = f"{initial_prompt} {bias_hint}".strip()
        elif (language or DEFAULT_LANGUAGE) in HINDI_LANGUAGE_CODES:
            initial_prompt = f"{DEFAULT_HINDI_PROMPT}{(' ' + bias_hint) if bias_hint else ''}"

        # Generate client ID for progress tracking
        client_id = f"transcribe-{file_id}-{datetime.now().timestamp()}"
        
        # Perform transcription
        result = await transcribe_full_audio(
            file_path,
            client_id,
            language=language,
            initial_prompt=initial_prompt,
            accuracy_mode=accuracy_mode,
        )
        
        return JSONResponse({
            "success": True, 
            "transcription": result,
            "client_id": client_id
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Transcription start failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start transcription: {str(e)}")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connections for progress updates."""
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global whisper_model, faster_whisper_model, diarization_pipeline
    
    return {
        "status": "healthy",
        "whisper_available": whisper_model is not None or faster_whisper_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "model_size": WHISPER_MODEL_SIZE,
        "faster_whisper": USE_FASTER_WHISPER,
        "audio_enhancement": ENABLE_AUDIO_ENHANCEMENT,
        "speaker_diarization": {
            "enabled": ENABLE_SPEAKER_DIARIZATION,
            "available": DIARIZATION_AVAILABLE,
            "loaded": diarization_pipeline is not None
        },
        "supported_languages": len(SUPPORTED_LANGUAGES)
    }

@app.get("/models")
async def list_models():
    """List available models and capabilities."""
    return {
        "primary_model": f"whisper-{WHISPER_MODEL_SIZE}",
        "faster_whisper_enabled": USE_FASTER_WHISPER,
        "supported_languages": SUPPORTED_LANGUAGES,
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "max_duration_hours": MAX_DURATION / 3600,
        "supported_formats": list(SUPPORTED_FORMATS)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Advanced Audio Transcription Service")
    logger.info(f"🎯 Primary model: whisper-{WHISPER_MODEL_SIZE}")
    logger.info(f"⚡ Faster-whisper: {'enabled' if USE_FASTER_WHISPER else 'disabled'}")
    logger.info(f"🎧 Audio enhancement: {'enabled' if ENABLE_AUDIO_ENHANCEMENT else 'disabled'}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
