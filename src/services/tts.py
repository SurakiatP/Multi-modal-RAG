"""
Text-to-Speech service using Edge TTS with fallback to gTTS
"""
import asyncio
import hashlib
from pathlib import Path
from typing import Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("Edge TTS not available, falling back to gTTS")

from gtts import gTTS

from ..config import TTS_CACHE_DIR, TTS_VOICE, TTS_LANGUAGE


class TTSService:
    """Text-to-Speech with caching"""
    
    def __init__(self):
        TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    async def _generate_edge_tts(self, text: str, output_path: Path) -> bool:
        """Generate speech using Edge TTS"""
        if not EDGE_TTS_AVAILABLE:
            return False
        
        try:
            communicate = edge_tts.Communicate(
                text,
                TTS_VOICE,
                rate="+5%"
            )
            await communicate.save(str(output_path))
            return True
        except Exception as e:
            print(f"Edge TTS error: {e}")
            return False
    
    def _generate_gtts(self, text: str, output_path: Path):
        """Generate speech using gTTS (fallback)"""
        try:
            tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=False)
            tts.save(str(output_path))
            return True
        except Exception as e:
            print(f"gTTS error: {e}")
            return False
    
    def text_to_speech(self, text: str) -> Optional[str]:
        """
        Convert text to speech with caching
        
        Args:
            text: Text to convert
            
        Returns:
            Path to audio file or None
        """
        if not text:
            return None
        
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_path = TTS_CACHE_DIR / f"{text_hash}.mp3"
        
        if cache_path.exists():
            print(f"Using cached TTS: {cache_path}")
            return str(cache_path)
        
        # Generate new
        if EDGE_TTS_AVAILABLE:
            success = asyncio.run(self._generate_edge_tts(text, cache_path))
            if success:
                print(f"Generated Edge TTS: {cache_path}")
                return str(cache_path)
        
        # Fallback to gTTS
        success = self._generate_gtts(text, cache_path)
        if success:
            print(f"Generated gTTS: {cache_path}")
            return str(cache_path)
        
        return None