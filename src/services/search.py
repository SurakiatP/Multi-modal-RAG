"""
RAG search and response generation
"""
from typing import List, Dict, Iterator, Tuple

from ..models.llm import LLMService
from ..database.qdrant_client import QdrantService
from ..services.tts import TTSService
from ..services.response import ResponseBuilder


class SearchService:
    """Handles search and response generation"""
    
    def __init__(self, qdrant_service: QdrantService, tts_service: TTSService):
        self.qdrant = qdrant_service
        self.tts = tts_service
        self.llm = LLMService()
        self.response_builder = ResponseBuilder()
    
    def search_and_respond(
        self,
        query: str,
        query_embedding,
        vector_name: str,
        llm_model: str
    ) -> Iterator[Tuple[str, List[str], str]]:
        """
        Search products and generate response
        
        Args:
            query: User query text
            query_embedding: Query vector
            vector_name: "text" or "image"
            llm_model: Ollama model name
            
        Yields:
            Tuple of (response_text, image_paths, audio_path)
        """
        # Search products
        retrieved_products = self.qdrant.search(
            query_embedding,
            vector_name=vector_name,
            limit=3
        )
        
        # Generate LLM response (streaming)
        llm_generator = self.llm.generate_response(
            query,
            retrieved_products,
            llm_model
        )
        
        # Build streaming response with images and audio
        for response_text, images, audio in self.response_builder.build_streaming_response(
            llm_generator,
            retrieved_products,
            tts_callback=self.tts.text_to_speech
        ):
            yield response_text, images, audio