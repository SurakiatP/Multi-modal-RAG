"""
LLM integration with Ollama
"""
import ollama
from typing import Iterator, Tuple, List

from ..config import AVAILABLE_OLLAMA_MODELS


class LLMService:
    """Handles LLM interactions with Ollama"""
    
    SYSTEM_PROMPT = """You are a helpful shop assistant in a retail store in Thailand. Your job is to help customers find products by answering questions about product names, prices, codes, and shelf locations.

You must always respond in **Thai language only**. Do not use English or any other language.

When a customer asks about a product, follow these rules:

1. **If the product is available in your database:**
   - Provide:
     - Product name (ชื่อสินค้า)
     - Product code (รหัสสินค้า)
     - Price (ราคา) หน่วยเป็น บาท
     - Shelf location (ตำแหน่งในร้าน เช่น ชั้นวาง A2 แถว 1)

2. **If the exact product is not found:**
   - First, say: "**ไม่พบสินค้า**"
   - Then, suggest up to 2 similar or related products (ถ้ามี) with their:
     - Product name
     - Price (บาท)
     - Location

3. **If no similar products are found:**
   - Simply reply: "**ไม่พบสินค้า**"

Important:
- Do not fabricate product information.
- Only suggest products that are actually available.
- Be polite and concise like a real store assistant.
- Price in Baht."""
    
    @staticmethod
    def generate_response(
        query: str,
        retrieved_products: List[dict],
        model_name: str
    ) -> Iterator[Tuple[str, List[str], str]]:
        """
        Generate streaming response from LLM
        
        Args:
            query: User query
            retrieved_products: Products from RAG search
            model_name: Ollama model name
            
        Yields:
            Tuple of (response_text, image_paths, audio_path)
        """
        # Build context
        context = ""
        if retrieved_products:
            context += "ข้อมูลสินค้าที่พบ:\n"
            for product in retrieved_products:
                context += (
                    f"ชื่อ: {product['name']}, "
                    f"รายละเอียด: {product['description']}, "
                    f"ราคา: {product['price']}, "
                    f"รหัส: {product['code']}, "
                    f"ตำแหน่ง: {product['location']}\n"
                )
        else:
            context += "ไม่พบสินค้าที่เกี่ยวข้อง."
        
        # Build messages
        messages = [
            {"role": "system", "content": LLMService.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"ผู้ใช้ถาม: {query}\n\nจากข้อมูลต่อไปนี้: {context}\n\n"
                          f"โปรดตอบคำถามของผู้ใช้และระบุรหัส,ราคารายละเอียดและตำแหน่งสินค้า"
            }
        ]
        
        full_response = ""
        
        try:
            # Stream from Ollama
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                options={'temperature': 0.5},
                stream=True
            )
            
            for chunk in stream:
                if 'content' in chunk['message']:
                    chunk_text = chunk['message']['content']
                    full_response += chunk_text
                    
                    # Yield intermediate result
                    yield full_response, [], None
            
            # Final yield with complete response
            yield full_response, [], None
            
        except Exception as e:
            error_msg = (
                f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Ollama: {e}\n"
                f"โปรดตรวจสอบว่า Ollama Server ทำงานอยู่และโมเดล '{model_name}' "
                f"ได้รับการดาวน์โหลดแล้ว"
            )
            yield error_msg, [], None