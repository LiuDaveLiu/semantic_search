import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import logging

from src.config import Config

logger = logging.getLogger(__name__)

class Reranker:
    """Cross-encoder for reranking top candidates"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.device = str(Config.DEVICE)
        self.batch_size = Config.RERANKER_BATCH_SIZE
        
        logger.info(f"Loading reranker {model_name} on {self.device}")
        self.model = CrossEncoder(model_name, device=self.device, max_length=512)
    
    def rerank(self, query: str, candidates: List[Tuple[str, str]], top_k: int = 10) -> List[str]:
        """Rerank candidates for a single query"""
        if not candidates:
            return []
        
        # Prepare pairs
        pairs = [[query, text] for _, text in candidates]
        
        # Score
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        # Sort and return top-k IDs
        scored = [(candidates[i][0], scores[i]) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [pid for pid, _ in scored[:top_k]]
    
    def rerank_batch(self, queries: List[str], 
                     candidates_batch: List[List[Tuple[str, str]]], 
                     top_k: int = 10) -> List[List[str]]:
        """Rerank multiple queries efficiently"""
        results = []
        
        for query, candidates in zip(queries, candidates_batch):
            reranked = self.rerank(query, candidates, top_k)
            results.append(reranked)
        
        return results