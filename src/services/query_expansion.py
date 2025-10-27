"""
Query expansion for better retrieval
"""
from typing import List


class QueryExpander:
    """Expand queries with synonyms"""
    
    def __init__(self):
        self.synonyms = {
            # หมวดหมู่อาหาร
            "ข้าวกล่อง": ["ข้าวกล่อง", "ข้าวกล่อง", "ข้าวกล่อง"],
            "อาหารพร้อมทาน": ["อาหารพร้อมทาน", "อาหาร", "ready to eat"],
            
            # หมวดหมู่ขนม
            "ขนม": ["ขนม", "ขนมขบเคี้ยว", "snack", "ของหวาน"],
            
            # เครื่องดื่ม
            "เครื่องดื่มเย็น": ["เครื่องดื่ม", "ของเย็น", "น้ำ", "drink"],
            "เครื่องดื่ม": ["เครื่องดื่ม", "น้ำ", "drink"],
            
            # ราคา
            "ราคาถูก": ["budget", "ราคาประหยัด", "ราคาไม่แพง"],
        }
    
    def expand(self, query: str) -> str:
        """Expand query with synonyms"""
        if query in self.synonyms:
            expanded = " ".join(self.synonyms[query])
            return expanded
        
        return query