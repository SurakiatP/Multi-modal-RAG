"""
Response generation and processing service
"""
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class ResponseProcessor:
    """Process and format chatbot responses"""
    
    @staticmethod
    def extract_product_codes(text: str) -> List[str]:
        """
        Extract product codes from text
        
        Args:
            text: Text to search for product codes
            
        Returns:
            List of product codes found
        """
        # Pattern for product codes (e.g., P001, PROD-123, etc.)
        pattern = r'\b[A-Z]+[0-9]+[A-Z0-9-]*\b'
        codes = re.findall(pattern, text, re.IGNORECASE)
        return [code.upper() for code in codes]
    
    @staticmethod
    def find_matched_products(
        llm_response: str,
        retrieved_products: List[Dict]
    ) -> List[Dict]:
        """
        Find products that are mentioned in LLM response
        
        Args:
            llm_response: Response from LLM
            retrieved_products: Products from vector search
            
        Returns:
            List of matched products
        """
        if not retrieved_products:
            return []
        
        matched = []
        product_codes = [p['code'] for p in retrieved_products]
        
        for code in product_codes:
            pattern = r'\b' + re.escape(code) + r'\b'
            if re.search(pattern, llm_response, re.IGNORECASE):
                for product in retrieved_products:
                    if product['code'].lower() == code.lower():
                        matched.append(product)
                        break
        
        return matched
    
    @staticmethod
    def format_product_response(product: Dict) -> Tuple[str, str]:
        """
        Format product information for display and speech
        
        Args:
            product: Product dictionary
            
        Returns:
            Tuple of (display_text, speech_text)
        """
        display_text = (
            f"\n\nสินค้าที่คุณหาคือ **{product['name']}** "
            f"ที่มีรายละเอียดว่า \"{product['description']}\" ค่ะ\n"
            f"รหัสสินค้า: **{product['code']}**\n"
            f"ราคา **{product['price']}** บาท\n"
            f"สินค้าอยู่ที่ **{product['location']}** ค่ะ\n"
        )
        
        speech_text = (
            f"สินค้าที่คุณหาคือ {product['name']} "
            f"รหัสสินค้า {product['code']} "
            f"ราคา {product['price']} บาท"
            f"สินค้าอยู่ที่ {product['location']} ค่ะ"
        )
        
        return display_text, speech_text
    
    @staticmethod
    def get_product_images(product: Dict) -> List[str]:
        """
        Get list of valid image paths from product
        
        Args:
            product: Product dictionary
            
        Returns:
            List of valid image paths
        """
        images = []
        
        if 'image_paths' not in product:
            return images
        
        for img_path in product['image_paths']:
            if Path(img_path).exists():
                images.append(img_path)
        
        return images
    
    @staticmethod
    def format_not_found_response(
        llm_response: str,
        retrieved_products: List[Dict]
    ) -> Tuple[str, str, List[str]]:
        """
        Format response when product is not found
        
        Args:
            llm_response: Response from LLM
            retrieved_products: Products from search
            
        Returns:
            Tuple of (display_text, speech_text, image_paths)
        """
        display_text = llm_response
        speech_text = ""
        images = []
        
        # Check if LLM already mentioned not found
        not_found_keywords = ["ไม่พบสินค้า", "ขออภัย", "ไม่มีสินค้า", "หาไม่เจอ"]
        llm_mentioned_not_found = any(
            keyword in llm_response 
            for keyword in not_found_keywords
        )
        
        if not llm_mentioned_not_found:
            # LLM didn't mention not found
            if retrieved_products:
                # We have products but LLM didn't mention them
                display_text += (
                    "\n\nเราพบสินค้าที่เกี่ยวข้องในร้าน "
                    "แต่ไม่แน่ใจว่าตรงตามที่คุณต้องการหรือไม่"
                )
                speech_text = (
                    "เราพบสินค้าที่เกี่ยวข้องในร้าน "
                    "แต่ไม่แน่ใจว่าตรงตามที่คุณต้องการหรือไม่"
                )
                
                # Show first product as suggestion
                first_product = retrieved_products[0]
                images = ResponseProcessor.get_product_images(first_product)
                
                if images:
                    display_text += " ตัวอย่างสินค้าที่ใกล้เคียง:\n"
            else:
                # No products found at all
                display_text += (
                    "\n\nขออภัยค่ะ "
                    "ไม่พบสินค้าที่คุณกำลังมองหาในร้านของเรา"
                )
                speech_text = (
                    "ขออภัยค่ะ "
                    "ไม่พบสินค้าที่คุณกำลังมองหาในร้านของเรา"
                )
        else:
            # LLM already mentioned not found, use its response
            speech_text = llm_response
        
        return display_text, speech_text, images
    
    @staticmethod
    def clean_response_text(text: str) -> str:
        """
        Clean and format response text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def validate_product_data(product: Dict) -> bool:
        """
        Validate that product has required fields
        
        Args:
            product: Product dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['code', 'name', 'description', 'price', 'location']
        
        for field in required_fields:
            if field not in product or not product[field]:
                return False
        
        return True
    
    @staticmethod
    def format_multiple_products(products: List[Dict]) -> str:
        """
        Format multiple products for display
        
        Args:
            products: List of products
            
        Returns:
            Formatted text
        """
        if not products:
            return ""
        
        text = "\n\nสินค้าที่เกี่ยวข้อง:\n"
        
        for i, product in enumerate(products[:3], 1):  # Limit to 3 products
            text += (
                f"\n{i}. **{product['name']}** "
                f"({product['code']})\n"
                f"   ราคา: {product['price']} บาท\n"
                f"   ตำแหน่ง: {product['location']}\n"
            )
        
        return text
    
    @staticmethod
    def extract_price_from_text(text: str) -> Optional[str]:
        """
        Extract price information from text
        
        Args:
            text: Text containing price
            
        Returns:
            Extracted price or None
        """
        # Pattern for Thai price format (e.g., 35 บาท, 35.50 บาท)
        pattern = r'(\d+(?:\.\d+)?)\s*บาท'
        match = re.search(pattern, text)
        
        if match:
            return match.group(0)
        
        return None
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitize user input text
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text
        """
        # Remove potentially harmful characters
        text = re.sub(r'[<>\"\'\\]', '', text)
        
        # Limit length
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()


class ResponseBuilder:
    """Build complete responses with text, images, and audio"""
    
    def __init__(self, response_processor: Optional[ResponseProcessor] = None):
        """
        Initialize response builder
        
        Args:
            response_processor: Optional ResponseProcessor instance
        """
        self.processor = response_processor or ResponseProcessor()
    
    def build_response(
        self,
        llm_response: str,
        retrieved_products: List[Dict],
        tts_callback=None
    ) -> Tuple[str, List[str], Optional[str]]:
        """
        Build complete response with text, images, and audio
        
        Args:
            llm_response: Response from LLM
            retrieved_products: Products from search
            tts_callback: Optional callback function for TTS
            
        Returns:
            Tuple of (response_text, image_paths, audio_path)
        """
        # Find matched products
        matched_products = self.processor.find_matched_products(
            llm_response,
            retrieved_products
        )
        
        response_text = llm_response
        images = []
        speech_text = ""
        
        if matched_products:
            # Product found and matched
            product = matched_products[0]
            
            # Validate product
            if not self.processor.validate_product_data(product):
                return self._build_error_response()
            
            # Format response
            display_text, speech_text = self.processor.format_product_response(
                product
            )
            response_text += display_text
            
            # Get images
            images = self.processor.get_product_images(product)
            
        else:
            # No matched products
            display_text, speech_text, images = (
                self.processor.format_not_found_response(
                    llm_response,
                    retrieved_products
                )
            )
            response_text = display_text
        
        # Clean response text
        response_text = self.processor.clean_response_text(response_text)
        
        # Generate audio if callback provided
        audio_path = None
        if tts_callback and speech_text:
            audio_path = tts_callback(speech_text)
        
        return response_text, images, audio_path
    
    def _build_error_response(self) -> Tuple[str, List[str], None]:
        """Build error response"""
        error_text = (
            "เกิดข้อผิดพลาดในการประมวลผลข้อมูลสินค้า "
            "กรุณาลองใหม่อีกครั้ง"
        )
        return error_text, [], None
    
    def build_streaming_response(
        self,
        llm_chunks,
        retrieved_products: List[Dict],
        tts_callback=None
    ):
        """
        Build streaming response from LLM chunks
        
        Args:
            llm_chunks: Iterator of LLM response chunks
            retrieved_products: Products from search
            tts_callback: Optional callback function for TTS
            
        Yields:
            Tuple of (response_text, image_paths, audio_path)
        """
        full_response = ""
        
        # Stream LLM response
        for chunk_response, _, _ in llm_chunks:
            full_response = chunk_response
            yield full_response, [], None
        
        # Build final response with images and audio
        final_response, images, audio = self.build_response(
            full_response,
            retrieved_products,
            tts_callback
        )
        
        yield final_response, images, audio