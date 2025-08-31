import torch
from pathlib import Path

class Config:
    """Global configuration"""
    
    # Paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    CACHE_DIR = ROOT_DIR / "cache"
    RESULTS_DIR = ROOT_DIR / "results"
    
    # Create directories
    for dir_path in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_FP16 = torch.cuda.is_available()
    
    # Model configurations (RTX 4060 optimized)
    MODEL_CONFIGS = {
        'all-MiniLM-L6-v2': {
            'batch_size': 384,
            'dimension': 384,
            'max_seq_length': 256
        },
        'all-MiniLM-L12-v2': {
            'batch_size': 256,
            'dimension': 384,
            'max_seq_length': 256
        },
        'multi-qa-MiniLM-L6-cos-v1': {
            'batch_size': 384,
            'dimension': 384,
            'max_seq_length': 256
        },
        'multi-qa-distilbert-cos-v1': {
            'batch_size': 256,
            'dimension': 768,
            'max_seq_length': 256
        },
        'all-mpnet-base-v2': {
            'batch_size': 128,  # Reduced for RTX 4060
            'dimension': 768,
            'max_seq_length': 384
        }
    }
    
    # Index configurations
    INDEX_CONFIGS = {
        'flat': {'type': 'Flat', 'params': {}},
        'ivf': {'type': 'IVF', 'params': {'nlist': 256}},
        'hnsw': {'type': 'HNSW', 'params': {'M': 16}}
    }
    
    # Reranker settings
    RERANKER_BATCH_SIZE = 64
    RERANKER_TOP_K = 50  # How many candidates to rerank
    
    # Text preparation (optimized based on results)
    MAX_TITLE_REPEATS = 2  # Repeat title for emphasis
    MAX_DESCRIPTION_LENGTH = 600  # Increased from 400
    MAX_BULLETS_LENGTH = 400  # Increased from 200
    INCLUDE_COLOR = True
    INCLUDE_BRAND_PREFIX = True  # Add brand at beginning