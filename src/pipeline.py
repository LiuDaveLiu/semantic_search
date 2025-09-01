import time
from typing import Dict, List, Optional, Any
import torch
import faiss
import pickle
from pathlib import Path

from src.config import Config
from src.data_loader import ESCIDataLoader
from src.embeddings import GPUEmbeddingEngine
from src.indexing import VectorIndex
from src.reranking import Reranker
from src.evaluation import Evaluator

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
        self.df_merged = self.data_loader.prepare_dataset(version, 'us', sample_frac)
        self.df_train, self.df_test = self.data_loader.get_splits()
        self.unique_products = self.data_loader.get_unique_products()
    
    def build_index(self, cache_key: Optional[str] = None):
        """Build product index"""
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
    
    def save_index(self, save_dir: Optional[Path] = None):
        """
        Save the FAISS index and related files
        
        Args:
            save_dir: Directory to save index files (default: ./indices/)
        """
        if save_dir is None:
            save_dir = Config.ROOT_DIR / "indices"
        
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectory for this configuration
        config_name = f"{self.text_strategy}_{self.config.get('embedding_model', '').split('/')[-1]}"
        index_dir = save_dir / config_name
        index_dir.mkdir(exist_ok=True)
        
        print(f"💾 Saving index to {index_dir}")
        
        # Save FAISS index
        index_path = index_dir / "index.faiss"
        faiss.write_index(self.index.index, str(index_path))
        
        # Save product IDs
        ids_path = index_dir / "product_ids.pkl"
        with open(ids_path, 'wb') as f:
            pickle.dump(self.index.product_ids, f)
        
        # Save product texts (needed for reranking)
        texts_path = index_dir / "product_texts.pkl"
        with open(texts_path, 'wb') as f:
            pickle.dump(self.product_texts, f)
        
        # Save configuration
        config_path = index_dir / "config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        # Report sizes
        total_size = 0
        for file_path in [index_path, ids_path, texts_path, config_path]:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {file_path.name}: {size_mb:.1f} MB")
        
        print(f"  Total: {total_size:.1f} MB")
        print(f"✅ Index saved successfully to {index_dir}")
        
        return index_dir
    
    def load_index(self, index_dir: Path):
        """
        Load a saved FAISS index
        
        Args:
            index_dir: Directory containing saved index files
        """
        print(f"📦 Loading index from {index_dir}")
        
        # Load FAISS index
        self.index.index = faiss.read_index(str(index_dir / "index.faiss"))
        
        # Load product IDs
        with open(index_dir / "product_ids.pkl", 'rb') as f:
            self.index.product_ids = pickle.load(f)
        
        # Load product texts
        with open(index_dir / "product_texts.pkl", 'rb') as f:
            self.product_texts = pickle.load(f)
        
        # Load configuration (optional)
        config_path = index_dir / "config.pkl"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                saved_config = pickle.load(f)
                print(f"  Loaded config: {saved_config.get('name', 'unnamed')}")
        
        print(f"✅ Loaded index with {self.index.index.ntotal} vectors")
    
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
    
    def run_full_pipeline(self, version: str = 'small', sample_frac: Optional[float] = None, save_index: bool = True):
        """Run complete pipeline end-to-end"""
        start_time = time.time()
        
        # Check if index already exists
        index_dir = Config.ROOT_DIR / "indices" / f"{self.text_strategy}_{self.config.get('embedding_model', '').split('/')[-1]}"
        
        if index_dir.exists() and (index_dir / "index.faiss").exists():
            print(f"📦 Found existing index at {index_dir}")
            self.load_index(index_dir)
            
            # Still need to load data for evaluation
            self.prepare_data(version, sample_frac)
        else:
            # Prepare data
            self.prepare_data(version, sample_frac)
            
            # Build index
            self.build_index()
            
            # Save index if requested
            if save_index:
                self.save_index()
        
        # Evaluate
        metrics = self.evaluate()
        
        # Add timing
        metrics['Total_Time'] = time.time() - start_time
        
        return metrics
