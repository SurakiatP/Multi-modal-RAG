"""
Embedding models for text and images
"""
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
from pythainlp.tokenize import word_tokenize
from pathlib import Path

from ..config import CLIP_MODEL_NAME, TEXT_EMBEDDING_MODEL


class EmbeddingModels:
    """Manages CLIP and text embedding models"""
    
    def __init__(self):
        print("Loading embedding models...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU: {gpu_name}")
            except:
                print("GPU: Available")
        
        # CLIP for images
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        
        # SentenceTransformer for text
        self.text_model = SentenceTransformer(
            TEXT_EMBEDDING_MODEL,
            device=self.device
        )
        
        # Move to GPU if available
        if self.device == "cuda":
            self.clip_model.to(self.device)
            print("Models loaded on GPU!")
        else:
            print("Models loaded on CPU")
        
        print("Embedding models loaded successfully!")
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def get_text_embedding(self, text: str, is_query: bool = True) -> np.ndarray:
        """Generate text embedding using E5 model"""
        prefix = "query: " if is_query else "passage: "
        
        # Tokenize Thai text
        is_thai = any('\u0E00' <= c <= '\u0E7F' for c in text)
        if is_thai:
            processed_text = " ".join(word_tokenize(text, keep_whitespace=False))
        else:
            processed_text = text
        
        final_text = prefix + processed_text
        embeddings = self.text_model.encode(
            final_text,
            convert_to_tensor=False,
            normalize_embeddings=True  # เพิ่ม normalization
        )
        
        return embeddings
    
    def get_image_embedding(self, image_paths: list) -> np.ndarray:
        """Generate image embedding using CLIP"""
        if not image_paths:
            return None
        
        # หารูปที่มีอยู่จริง
        valid_image_path = None
        for img_path in image_paths:
            if Path(img_path).exists():
                valid_image_path = img_path
                break
        
        if valid_image_path is None:
            print(f"  Warning: No valid images found in {image_paths}")
            return None
        
        try:
            image = Image.open(valid_image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Move to GPU
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                embeddings = self.clip_model.get_image_features(**inputs)
            
            # Normalize
            embedding = embeddings.squeeze().cpu().numpy()
            return self.normalize_embedding(embedding)
            
        except Exception as e:
            print(f"  Warning: Error processing image {valid_image_path}: {e}")
            return None