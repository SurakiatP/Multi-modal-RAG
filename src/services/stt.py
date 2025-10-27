"""
Speech-to-Text service using Faster-Whisper
"""
import torch
from faster_whisper import WhisperModel
from typing import Optional

from ..config import WHISPER_MODEL_SIZE


class STTService:
    """Speech-to-Text using Faster-Whisper"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load Faster-Whisper model"""
        if self.model is not None:
            return
        
        print(f"Loading Faster-Whisper model '{WHISPER_MODEL_SIZE}'...")
        
        try:
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                print(f"Using GPU for Whisper")
            else:
                device = "cpu"
                compute_type = "int8"
                print("Using CPU for Whisper")
            
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=device,
                compute_type=compute_type
            )
            
            print(f"Whisper model loaded on {device}")
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Trying CPU fallback...")
            try:
                self.model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8"
                )
                print("Whisper model loaded on CPU (fallback)")
            except Exception as e2:
                print(f"CPU fallback also failed: {e2}")
                self.model = "fallback"
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio to text"""
        if self.model == "fallback":
            return None
        
        if self.model is None:
            self.load_model()
            if self.model == "fallback":
                return None
        
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language='th',
                beam_size=5,
                vad_filter=True
            )
            
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None