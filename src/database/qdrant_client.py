"""
Qdrant vector database operations
"""
import uuid
import numpy as np
import time
from qdrant_client import QdrantClient, models
from typing import List, Dict

from ..config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, QDRANT_TIMEOUT
from .product_loader import ProductLoader


class QdrantService:
    """Manages Qdrant vector database"""
    
    def __init__(self, embedding_models):
        """
        Initialize Qdrant service
        
        Args:
            embedding_models: Instance of EmbeddingModels
        """
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        
        # Retry connection with timeout
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    timeout=60
                )
                
                self.client.get_collections()
                print("Connected to Qdrant successfully!")
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(
                        f"Failed to connect to Qdrant after {max_retries} attempts. "
                        f"Please ensure Qdrant is running on {QDRANT_HOST}:{QDRANT_PORT}. "
                        f"Error: {e}"
                    )
        
        self.embedding_models = embedding_models
        self.collection_name = COLLECTION_NAME
    
    def initialize_collection(self):
        """Initialize Qdrant collection with product data"""
        try:
            # Delete existing collection
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
                time.sleep(2)
            
            # Create new collection
            print(f"Creating collection: {self.collection_name}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text": models.VectorParams(size=768, distance=models.Distance.COSINE),
                    "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
                },
                timeout=60
            )
            print(f"Created collection: {self.collection_name}")
            time.sleep(1)
            
            # Load and embed products
            products = ProductLoader.get_all_products()
            
            if not products:
                print("Warning: No products found to embed!")
                return
            
            print(f"Embedding {len(products)} products...")
            
            for i, product in enumerate(products, 1):
                print(f"  Processing product {i}/{len(products)}: {product.get('name', 'Unknown')}")
                self._add_product(product)
            
            # Create payload index
            print("Creating payload index...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="name",
                field_schema="keyword",
            )
            
            print("Qdrant initialization complete!")
            
        except Exception as e:
            print(f"Error during Qdrant initialization: {e}")
            raise
    
    def _add_product(self, product: Dict):
        """Add single product to Qdrant"""
        try:
            text_parts = [
                product['name'],
                product['name'],
                product['description'],
            ]
            
            if 'tags' in product and product['tags']:
                important_tags = product['tags'][:5]
                text_parts.extend(important_tags)
            
            if 'price_range' in product:
                text_parts.append(product['price_range'])
            
            combined_text = " ".join(text_parts)
            
            # Generate embeddings
            text_embedding = self.embedding_models.get_text_embedding(
                combined_text,
                is_query=False
            )
            
            # Image embedding
            image_embedding = self.embedding_models.get_image_embedding(
                product.get('image_paths', [])
            )
            
            if image_embedding is None:
                image_embedding = np.zeros(512)
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "text": text_embedding.tolist(),
                            "image": image_embedding.tolist(),
                        },
                        payload=product
                    )
                ],
                wait=True
            )
            
        except Exception as e:
            print(f"Error adding product {product.get('name', 'Unknown')}: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        vector_name: str = "text",
        limit: int = 3
    ) -> List[Dict]:
        """
        Search products in Qdrant
        
        Args:
            query_embedding: Query vector
            vector_name: "text" or "image"
            limit: Number of results
            
        Returns:
            List of product dictionaries
        """
        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                using=vector_name,
                limit=limit,
                with_payload=True,
            )
            
            return [hit.payload for hit in search_result.points]
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []