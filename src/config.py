"""
Configuration file for Robot Store
"""
import os
import torch
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
TTS_CACHE_DIR = CACHE_DIR / "tts_cache"
QDRANT_STORAGE_DIR = CACHE_DIR / "qdrant_storage"

# Data paths
PRODUCTS_JSON = DATA_DIR / "products.json"
IMAGES_DIR = DATA_DIR / "images"

# Qdrant settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "products"
QDRANT_TIMEOUT = 30

# Device settings
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

# GPU Memory management
if USE_GPU:
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# Model settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
WHISPER_MODEL_SIZE = "small"

# Ollama models
AVAILABLE_OLLAMA_MODELS = ["llama3.2:1b", "gemma2:2b", "qwen2.5:0.5b"]

# TTS settings
TTS_VOICE = "th-TH-PremwadeeNeural"
TTS_LANGUAGE = "th"

# Audio settings
AUDIO_RESPONSE_PATH = TTS_CACHE_DIR / "response.mp3"

# Create necessary directories
for directory in [DATA_DIR, IMAGES_DIR, CACHE_DIR, TTS_CACHE_DIR, QDRANT_STORAGE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Print device info
if USE_GPU:
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Mode: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    except:
        print("GPU Mode: Enabled")
else:
    print("CPU Mode")