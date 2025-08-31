import time
from typing import Dict, List, Optional, Any
import logging
import torch
import gc

from src.config import Config
from src.data_loader import ESCIDataLoader
from src.embeddings import GPUEmbeddingEngine
from src.indexing import VectorIndex
from src.reranking import Reranker
from src.evaluation import Evaluator

logger = logging.getLogger(__name__)

class SemanticSearchPipeline:
    """Complete semantic search pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary with keys:
                - embedding_model: Model name for embeddings
                - index_type: Type of FAISS index
                - use_reranker: Whether to use reranking
                - reranker_model: Model for reranking
                - text_strategy: 'basic' or 'enhanced'
        """
        self.config = config
        self.data_loader = ESCIDataLoader()
        
        # Text preparation strategy
        self.text_strategy = config.get('text_strategy', 'enhanced')
        
        # Initialize components
        self.embedding_engine = GPUEmbeddingEngine(
            config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        
        self.index = VectorIndex(
            dimension=self.embedding_engine.get_dimension(),
            index_type=config.get('index_type', 'ivf')
        )
        
        self.reranker = None
        if config.get('use_reranker', False):
            self.reranker = Reranker(
                config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            )
        
        self.evaluator = Evaluator()
        self.product_texts = {}
    
    def prepare_data(self, version: str = 'small', sample_frac: Optional[float] = None):
        """Load and prepare data"""
        logger.info("Preparing data...")
        self.df_merged = self.data_loader.prepare_dataset(version, 'us', sample_frac)
        self.df_train, self.df_test = self.data_loader.get_splits()
        self.unique_products = self.data_loader.get_unique_products()
    
    def build_index(self, cache_key: Optional[str] = None):
        """Build product index"""
        logger.info(f"Building index with {self.text_strategy} text strategy...")
        
        # Prepare texts with selected strategy
        self.product_texts = self.data_loader.prepare_product_texts(
            self.unique_products,
            strategy=self.text_strategy
        )
        
        # Get embeddings
        product_ids = list(self.product_texts.keys())
        texts = list(self.product_texts.values())
        
        # Cache key includes strategy
        cache_key = cache_key or f"{self.text_strategy}_products_{len(texts)}"
        embeddings = self.embedding_engine.encode_products(texts, cache_key)
        
        # Add to index
        self.index.add(embeddings, product_ids)
    
    def search(self, queries: List[str], k: int = 10) -> List[List[str]]:
        """Search for products"""
        # Encode queries
        query_embeddings = self.embedding_engine.encode_queries(queries)
        
        # Search (retrieve more if using reranker)
        _, results = self.index.search(
            query_embeddings, 
            k=Config.RERANKER_TOP_K if self.reranker else k
        )
        
        # Rerank if enabled
        if self.reranker:
            candidates_batch = []
            for products in results:
                candidates = [(pid, self.product_texts.get(pid, "")) for pid in products]
                candidates_batch.append(candidates)
            
            results = self.reranker.rerank_batch(queries, candidates_batch, k)
        
        return results
    
    def evaluate(self, batch_size: int = 50) -> Dict[str, float]:
        """Evaluate on test set"""
        logger.info("Evaluating...")
        
        # Get test data
        test_queries = self.data_loader.get_test_queries(self.df_test)
        ground_truth = self.data_loader.get_ground_truth(self.df_test)
        
        # Process in batches
        all_predictions = {}
        query_ids = list(test_queries.keys())
        query_texts = list(test_queries.values())
        
        for i in range(0, len(query_texts), batch_size):
            batch_ids = query_ids[i:i+batch_size]
            batch_texts = query_texts[i:i+batch_size]
            
            results = self.search(batch_texts, k=10)
            
            for qid, res in zip(batch_ids, results):
                all_predictions[qid] = res
            
            # Clear GPU memory periodically
            if torch.cuda.is_available() and i % 200 == 0:
                torch.cuda.empty_cache()
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(all_predictions, ground_truth)
        return metrics
    
    def run_full_pipeline(self, version: str = 'small', sample_frac: Optional[float] = None):
        """Run complete pipeline end-to-end"""
        start_time = time.time()
        
        # Prepare data
        self.prepare_data(version, sample_frac)
        
        # Build index
        self.build_index()
        
        # Evaluate
        metrics = self.evaluate()
        
        # Add timing
        metrics['Total_Time'] = time.time() - start_time
        
        return metrics
