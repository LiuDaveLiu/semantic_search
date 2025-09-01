import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import gc
from pathlib import Path
import pickle
import hashlib
import time
from tqdm.auto import tqdm

from src.config import Config

class GPUEmbeddingEngine:
    """GPU-accelerated embedding generation with persistent caching"""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 cache_dir: Optional[Path] = None):
        self.model_name = model_name
        self.device = device or str(Config.DEVICE)
        self.cache_dir = cache_dir or Config.CACHE_DIR
        
        # Model-specific cache directory
        model_cache_name = model_name.replace('/', '_').replace('-', '_')
        self.model_cache_dir = self.cache_dir / model_cache_name
        self.model_cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Get configuration
        model_key = model_name.split('/')[-1]
        self.config = Config.MODEL_CONFIGS.get(model_key, {
            'batch_size': 128,
            'dimension': 768,
            'max_seq_length': 256
        })
        
        self.model.max_seq_length = self.config['max_seq_length']
        self.batch_size = self.config['batch_size']
        
        self._print_cache_status()
    
    def _get_cache_key(self, texts: List[str], prefix: str = "embeddings") -> str:
        """Generate unique cache key"""
        # Include text strategy in hash
        hash_content = f"{self.model_name}_{len(texts)}_{texts[0] if texts else ''}_{texts[-1] if texts else ''}"
        content_hash = hashlib.md5(hash_content.encode()).hexdigest()[:12]
        return f"{prefix}_{len(texts)}_{content_hash}"
    
    def _print_cache_status(self):
        """Print cache status"""
        cache_files = list(self.model_cache_dir.glob("*.pkl"))
        if cache_files:
            total_size = sum(f.stat().st_size for f in cache_files) / (1024**3)
            print(f"📦 Found {len(cache_files)} cached embeddings ({total_size:.2f} GB)")
        else:
            print(f"📦 No cached embeddings for {self.model_name}")
    
    def encode(self, 
               texts: List[str],
               batch_size: Optional[int] = None,
               show_progress: bool = True,
               use_cache: bool = True,
               cache_key: Optional[str] = None,
               normalize: bool = True) -> np.ndarray:
        """Encode texts with caching"""
        if not texts:
            return np.array([])
        
        # Check cache
        if cache_key is None:
            cache_key = self._get_cache_key(texts)
        
        cache_path = self.model_cache_dir / f"{cache_key}.pkl"
        
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Generate embeddings
        batch_size = batch_size or self.batch_size
        
        # Clear GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        start_time = time.time()
        
        try:
            # Use mixed precision on GPU
            if self.device == 'cuda' and Config.USE_FP16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=show_progress,
                        device=self.device,
                        normalize_embeddings=normalize,
                        convert_to_numpy=True
                    )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True
                )
            
            encoding_time = time.time() - start_time
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()
                
                return self.encode(texts, batch_size=batch_size//2, show_progress=show_progress,
                                 use_cache=use_cache, cache_key=cache_key, normalize=normalize)
            else:
                raise e
        
        # Save to cache
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f, protocol=4)
        
        # Clear GPU memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return embeddings
    
    def encode_products(self, texts: List[str], cache_key: str = 'products') -> np.ndarray:
        """Encode products with automatic caching"""
        return self.encode(texts, use_cache=True, cache_key=cache_key)
    
    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Encode queries without caching"""
        return self.encode(texts, use_cache=False, show_progress=False)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.config['dimension']
    
    def clear_cache(self):
        """Clear cache for this model"""
        cache_files = list(self.model_cache_dir.glob("*.pkl"))
        for f in cache_files:
            f.unlink()