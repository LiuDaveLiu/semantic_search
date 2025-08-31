import faiss
import numpy as np
from typing import List, Tuple, Optional
import logging

from src.config import Config

logger = logging.getLogger(__name__)

class VectorIndex:
    """FAISS vector index for similarity search"""
    
    def __init__(self, dimension: int, index_type: str = 'ivf'):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.product_ids = []
        
        self.config = Config.INDEX_CONFIGS.get(index_type, {'type': 'IVF', 'params': {}})
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index"""
        logger.info(f"Creating {self.index_type} index (dim={self.dimension})")
        
        if self.config['type'] == 'Flat':
            # Exact search with cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            
        elif self.config['type'] == 'IVF':
            # Inverted file index for faster search
            nlist = self.config['params'].get('nlist', 256)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
        elif self.config['type'] == 'HNSW':
            # Hierarchical NSW for very fast search
            M = self.config['params'].get('M', 16)
            self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
    
    def add(self, embeddings: np.ndarray, ids: List[str]):
        """Add embeddings to index"""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Ensure normalization for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train if needed (for IVF)
        if hasattr(self.index, 'train') and not self.index.is_trained:
            logger.info("Training index...")
            train_size = min(len(embeddings), 50000)
            self.index.train(embeddings[:train_size])
        
        # Add embeddings
        logger.info(f"Adding {len(embeddings):,} vectors to index")
        self.index.add(embeddings)
        self.product_ids.extend(ids)
    
    def search(self, query_embeddings: np.ndarray, k: int = 10, 
              nprobe: Optional[int] = None) -> Tuple[np.ndarray, List[List[str]]]:
        """Search for nearest neighbors"""
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Normalize queries for cosine similarity
        faiss.normalize_L2(query_embeddings)
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe or min(32, self.index.nlist // 4)
        
        # Search
        distances, indices = self.index.search(query_embeddings, k)
        
        # Convert to product IDs
        results = []
        for query_indices in indices:
            query_results = [self.product_ids[idx] for idx in query_indices 
                           if 0 <= idx < len(self.product_ids)]
            results.append(query_results)
        
        return distances, results