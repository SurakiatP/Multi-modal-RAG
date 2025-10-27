"""
Comprehensive System Evaluation
"""
import time
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, ndcg_score

from src.models.embeddings import EmbeddingModels
from src.database.qdrant_client import QdrantService
from src.services.stt import STTService
from src.services.query_expansion import QueryExpander


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, embedding_models, qdrant_service):
        self.embedding_models = embedding_models
        self.qdrant_service = qdrant_service
        self.test_queries = self.load_test_queries()
        self.query_expander = QueryExpander()  
    
    def load_test_queries(self) -> List[Dict]:
        """Load test queries with ground truth"""
        return [
            {
                "query": "ข้าวกล่อง",
                "expected_codes": ["FD004", "FD005"],
                "category": "text"
            },
            {
                "query": "อาหารพร้อมทาน",
                "expected_codes": ["FD004", "FD005", "FD006", "FD007"],
                "category": "text"
            },
            {
                "query": "เครื่องดื่มเย็น",
                "expected_codes": ["DR005", "DR007", "DR008"],
                "category": "text"
            },
            {
                "query": "ขนม",
                "expected_codes": ["SN006", "SN007", "SN008", "BK004"],
                "category": "text"
            },
            {
                "query": "ราคาถูก",
                "expected_codes": ["DR008", "BK004", "SN008"],
                "category": "text"
            }
        ]
    
    def calculate_precision_at_k(
        self,
        retrieved: List[str],
        expected: List[str],
        k: int = 3
    ) -> float:
        """Calculate Precision@K"""
        retrieved_k = retrieved[:k]
        relevant = sum(1 for code in retrieved_k if code in expected)
        return relevant / k if k > 0 else 0
    
    def calculate_recall_at_k(
        self,
        retrieved: List[str],
        expected: List[str],
        k: int = 3
    ) -> float:
        """Calculate Recall@K"""
        retrieved_k = retrieved[:k]
        relevant = sum(1 for code in retrieved_k if code in expected)
        total_relevant = len(expected)
        return relevant / total_relevant if total_relevant > 0 else 0
    
    def calculate_mrr(
        self,
        retrieved: List[str],
        expected: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, code in enumerate(retrieved, 1):
            if code in expected:
                return 1.0 / i
        return 0.0
    
    def evaluate_retrieval(self, k: int = 3) -> Dict:
        """Evaluate retrieval metrics"""
        print("\n" + "="*60)
        print("RETRIEVAL EVALUATION")
        print("="*60)
        
        print("\nWarming up GPU...")
        warmup_embedding = self.embedding_models.get_text_embedding("test", is_query=True)
        _ = self.qdrant_service.search(warmup_embedding, vector_name="text", limit=1)
        print("Warm-up complete!\n")

        precisions = []
        recalls = []
        mrrs = []
        latencies = []
        
        for test_case in self.test_queries:
            query = test_case["query"]
            expected = test_case["expected_codes"]
            
            expanded_query = self.query_expander.expand(query)

            # Measure latency
            start_time = time.time()
            
            # Get embedding
            query_embedding = self.embedding_models.get_text_embedding(
                expanded_query,
                is_query=True
            )
            
            # Search
            results = self.qdrant_service.search(
                query_embedding,
                vector_name="text",
                limit=k
            )
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Extract codes
            retrieved_codes = [r['code'] for r in results]
            
            # Calculate metrics
            precision = self.calculate_precision_at_k(retrieved_codes, expected, k)
            recall = self.calculate_recall_at_k(retrieved_codes, expected, k)
            mrr = self.calculate_mrr(retrieved_codes, expected)
            
            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(mrr)
            
            # Print results
            print(f"\nQuery: '{query}'")
            print(f"  Expected: {expected}")
            print(f"  Retrieved: {retrieved_codes}")
            print(f"  Precision@{k}: {precision:.3f}")
            print(f"  Recall@{k}: {recall:.3f}")
            print(f"  MRR: {mrr:.3f}")
            print(f"  Latency: {latency:.3f}s")
        
        # Average metrics
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_mrr = np.mean(mrrs)
        avg_latency = np.mean(latencies)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Precision status
        precision_status = "PASS" if avg_precision > 0.7 else "WARN" if avg_precision > 0.6 else "FAIL"
        print(f"Average Precision@{k}: {avg_precision:.3f} [{precision_status}]")
        
        # Recall status
        recall_status = "PASS" if avg_recall > 0.7 else "WARN" if avg_recall > 0.6 else "FAIL"
        print(f"Average Recall@{k}: {avg_recall:.3f} [{recall_status}]")
        
        # MRR status
        mrr_status = "PASS" if avg_mrr > 0.8 else "WARN" if avg_mrr > 0.5 else "FAIL"
        print(f"Average MRR: {avg_mrr:.3f} [{mrr_status}]")
        
        # Latency status
        latency_status = "PASS" if avg_latency < 1 else "WARN" if avg_latency < 3 else "FAIL"
        print(f"Average Latency: {avg_latency:.3f}s [{latency_status}]")
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "mrr": avg_mrr,
            "latency": avg_latency
        }
    
    def evaluate_embedding_quality(self):
        """Evaluate embedding quality"""
        print("\n" + "="*60)
        print("EMBEDDING QUALITY EVALUATION")
        print("="*60)
        
        # Test similar items
        similar_pairs = [
            ("ข้าวกล่องไก่กระเทียม", "ข้าวผัดกะเพรา"),
            ("กาแฟ", "ชาเขียว"),
            ("ขนมเวเฟอร์", "เค้กกล้วยหอม")
        ]
        
        # Test dissimilar items
        dissimilar_pairs = [
            ("ข้าวกล่อง", "น้ำดื่ม"),
            ("กาแฟ", "ไส้กรอก"),
            ("เค้ก", "น้ำดื่ม")
        ]
        
        similar_scores = []
        dissimilar_scores = []
        
        print("\nSimilar Items (should have high similarity):")
        for item1, item2 in similar_pairs:
            emb1 = self.embedding_models.get_text_embedding(item1, is_query=False)
            emb2 = self.embedding_models.get_text_embedding(item2, is_query=False)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similar_scores.append(similarity)
            print(f"  {item1} <-> {item2}: {similarity:.3f}")
        
        print("\nDissimilar Items (should have low similarity):")
        for item1, item2 in dissimilar_pairs:
            emb1 = self.embedding_models.get_text_embedding(item1, is_query=False)
            emb2 = self.embedding_models.get_text_embedding(item2, is_query=False)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            dissimilar_scores.append(similarity)
            print(f"  {item1} <-> {item2}: {similarity:.3f}")
        
        avg_similar = np.mean(similar_scores)
        avg_dissimilar = np.mean(dissimilar_scores)
        separation = avg_similar - avg_dissimilar
        
        print(f"\nAverage Similar Score: {avg_similar:.3f}")
        print(f"Average Dissimilar Score: {avg_dissimilar:.3f}")
        
        separation_status = "PASS" if separation > 0.2 else "WARN" if separation > 0.1 else "FAIL"
        print(f"Separation: {separation:.3f} [{separation_status}]")
        
        return {
            "similar_score": avg_similar,
            "dissimilar_score": avg_dissimilar,
            "separation": separation
        }


def main():
    """Run evaluation"""
    print("="*60)
    print("RAG SYSTEM EVALUATION")
    print("="*60)
    
    # Initialize
    print("\nInitializing system...")
    embedding_models = EmbeddingModels()
    qdrant_service = QdrantService(embedding_models)
    
    # Create evaluator
    evaluator = RAGEvaluator(embedding_models, qdrant_service)
    
    # Run evaluations
    retrieval_metrics = evaluator.evaluate_retrieval(k=3)
    embedding_metrics = evaluator.evaluate_embedding_quality()
    
    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    
    passed = 0
    total = 0
    
    # Check retrieval
    if retrieval_metrics["precision"] > 0.7:
        print("[PASS] Precision: {:.3f}".format(retrieval_metrics["precision"]))
        passed += 1
    else:
        print("[FAIL] Precision: {:.3f}".format(retrieval_metrics["precision"]))
    total += 1
    
    if retrieval_metrics["recall"] > 0.7:
        print("[PASS] Recall: {:.3f}".format(retrieval_metrics["recall"]))
        passed += 1
    else:
        print("[FAIL] Recall: {:.3f}".format(retrieval_metrics["recall"]))
    total += 1
    
    if retrieval_metrics["mrr"] > 0.8:
        print("[PASS] MRR: {:.3f}".format(retrieval_metrics["mrr"]))
        passed += 1
    else:
        print("[FAIL] MRR: {:.3f}".format(retrieval_metrics["mrr"]))
    total += 1
    
    if retrieval_metrics["latency"] < 3.0:
        print("[PASS] Latency: {:.3f}s".format(retrieval_metrics["latency"]))
        passed += 1
    else:
        print("[FAIL] Latency: {:.3f}s".format(retrieval_metrics["latency"]))
    total += 1
    
    if embedding_metrics["separation"] > 0.2:
        print("[PASS] Embedding Quality: {:.3f}".format(embedding_metrics["separation"]))
        passed += 1
    else:
        print("[FAIL] Embedding Quality: {:.3f}".format(embedding_metrics["separation"]))
    total += 1
    
    print(f"\nFinal Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n[SUCCESS] System PASSED all tests!")
    elif passed >= total * 0.8:
        print("\n[WARNING] System needs minor improvements")
    else:
        print("\n[ERROR] System needs major improvements")
    
    # Save results - แก้ตรงนี้
    results = {
        "retrieval_metrics": {
            "precision": float(retrieval_metrics["precision"]),
            "recall": float(retrieval_metrics["recall"]),
            "mrr": float(retrieval_metrics["mrr"]),
            "latency": float(retrieval_metrics["latency"])
        },
        "embedding_metrics": {
            "similar_score": float(embedding_metrics["similar_score"]),
            "dissimilar_score": float(embedding_metrics["dissimilar_score"]),
            "separation": float(embedding_metrics["separation"])
        },
        "score": f"{passed}/{total}",
        "percentage": float(passed/total*100)
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to: evaluation_results.json")


if __name__ == "__main__":
    main()