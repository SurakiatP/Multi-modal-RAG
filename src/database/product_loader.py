"""
Product data loader from JSON
"""
import json
from pathlib import Path
from typing import List, Dict

from ..config import PRODUCTS_JSON


class ProductLoader:
    """Loads and manages product data"""
    
    _products_cache = None
    
    @classmethod
    def load_products(cls) -> List[Dict]:
        """
        Load products from JSON file with caching
        
        Returns:
            List of product dictionaries
        """
        if cls._products_cache is not None:
            return cls._products_cache
        
        if not PRODUCTS_JSON.exists():
            print(f"Error: Products file not found at '{PRODUCTS_JSON}'")
            return []
        
        try:
            with open(PRODUCTS_JSON, 'r', encoding='utf-8') as f:
                cls._products_cache = json.load(f)
            
            print(f"Successfully loaded {len(cls._products_cache)} products")
            return cls._products_cache
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error loading products: {e}")
            return []
    
    @classmethod
    def get_all_products(cls) -> List[Dict]:
        """Get all products"""
        return cls.load_products()
    
    @classmethod
    def reload_products(cls):
        """Force reload products from file"""
        cls._products_cache = None
        return cls.load_products()