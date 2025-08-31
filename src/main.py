"""
Semantic Product Search Solution
Author: Dave Liu
Date: August 2025
Hardware: Dell XPS 15 9530, 32GB RAM
Python: 3.12.8

This solution is optimized to leverage your hardware capabilities for better performance.
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Python 3.12 compatibility
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core dependencies
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure for your hardware
HARDWARE_CONFIG = {
    'cpu_cores': mp.cpu_count(),  # Should be 14 cores (6P+8E) for Intel Core i7-13700H
    'ram_gb': 32,
    'batch_size': 128,  # Increased from 32 due to more RAM
    'num_workers': 8,   # Parallel processing workers
    'cache_size_gb': 4  # Memory cache for embeddings
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('semantic_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# ============================================================================
# Performance Optimized Data Loader
# ============================================================================

@dataclass
class OptimizedDataLoader:
    """Hardware-optimized data loader leveraging 32GB RAM"""
    
    data_path: Path = field(default_factory=lambda: Path("./data/"))
    cache_embeddings: bool = True
    use_memory_mapping: bool = True
    
    def __post_init__(self):
        self.df_examples = None
        self.df_products = None
        self.df_merged = None
        self.embedding_cache = {}
        
        # Create necessary directories
        self.data_path.mkdir(exist_ok=True)
        (self.data_path / "cache").mkdir(exist_ok=True)
        
    def load_data(self, use_small_version: bool = False) -> pd.DataFrame:
        """
        Load ESCI dataset with optimizations for 32GB RAM
        Note: With 32GB RAM, we can easily handle the large version
        """
        logger.info(f"Loading ESCI dataset (large version: {not use_small_version})")
        
        # File paths
        examples_path = self.data_path / 'shopping_queries_dataset_examples.parquet'
        products_path = self.data_path / 'shopping_queries_dataset_products.parquet'
        
        # Load with optimized settings for parquet
        self.df_examples = pd.read_parquet(
            examples_path,
            engine='pyarrow',  # Faster for large files
            columns=None  # Load all columns
        )
        
        self.df_products = pd.read_parquet(
            products_path,
            engine='pyarrow'
        )
        
        # Efficient merge using categorical dtypes for repeated values
        for col in ['product_locale', 'esci_label', 'split']:
            if col in self.df_examples.columns:
                self.df_examples[col] = self.df_examples[col].astype('category')
        
        # Merge with memory optimization
        logger.info("Merging product and example data...")
        self.df_merged = pd.merge(
            self.df_examples,
            self.df_products,
            how='left',
            on=['product_locale', 'product_id'],
            copy=False  # Avoid unnecessary copying
        )
        
        # Filter version - with 32GB RAM, use large version for better results
        version_col = 'small_version' if use_small_version else 'large_version'
        self.df_merged = self.df_merged[self.df_merged[version_col] == 1]
        
        # Focus on English locale (can handle all locales with 32GB RAM if needed)
        self.df_merged = self.df_merged[self.df_merged['product_locale'] == 'us']
        
        # Optimize memory usage
        self.df_merged = self._optimize_dataframe_memory(self.df_merged)
        
        logger.info(f"Loaded {len(self.df_merged):,} query-product pairs")
        logger.info(f"Unique queries: {self.df_merged['query_id'].nunique():,}")
        logger.info(f"Unique products: {self.df_merged['product_id'].nunique():,}")
        logger.info(f"Memory usage: {self.df_merged.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        return self.df_merged
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                if col_type.name == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif col_type.name == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def prepare_product_texts_parallel(self, df_products: pd.DataFrame, 
                                      num_workers: int = 8) -> Dict[str, str]:
        """Prepare product texts using parallel processing"""
        logger.info(f"Preparing product texts with {num_workers} workers...")
        
        def process_row(row):
            # Combine fields with strategic weighting
            title = str(row['product_title']) if pd.notna(row['product_title']) else ''
            brand = str(row['product_brand']) if pd.notna(row['product_brand']) else ''
            description = str(row['product_description']) if pd.notna(row['product_description']) else ''
            bullets = str(row['product_bullet_point']) if pd.notna(row['product_bullet_point']) else ''
            color = str(row['product_color']) if pd.notna(row['product_color']) else ''
            
            # Smart truncation for better context
            description = description[:600]
            bullets = bullets[:400]
            
            # Enhanced text combination
            text_parts = []
            if title:
                text_parts.append(f"{title}")  # Title gets priority
            if brand and brand not in title:
                text_parts.append(f"by {brand}")
            if color and color not in title:
                text_parts.append(f"Color: {color}")
            if description:
                text_parts.append(f"{description}")
            if bullets and bullets not in description:
                text_parts.append(f"Features: {bullets}")
            
            return row['product_id'], " ".join(text_parts)
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_row, [row for _, row in df_products.iterrows()]),
                total=len(df_products),
                desc="Processing products"
            ))
        
        return dict(results)

# ============================================================================
# Enhanced Embedding Manager with Caching
# ============================================================================

class EnhancedEmbeddingManager:
    """Embedding manager optimized for 32GB RAM with caching"""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', 
                 device: str = 'cpu', cache_dir: Path = Path("./data/cache/")):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load sentence transformer with optimizations"""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(
            f'sentence-transformers/{self.model_name}',
            device=self.device
        )
        # Enable multi-process encoding for CPU
        self.model.max_seq_length = 512  # Increase for better context
        
    def encode_batch_optimized(self, texts: List[str], 
                              batch_size: int = 128,
                              show_progress: bool = True) -> np.ndarray:
        """Optimized batch encoding leveraging more RAM"""
        
        # Check cache first
        cache_key = f"{self.model_name}_{hash(tuple(texts[:10]))}.npy"
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists() and len(texts) > 1000:
            logger.info(f"Loading embeddings from cache: {cache_path}")
            return np.load(cache_path)
        
        # Encode with larger batches (more RAM available)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Pre-normalize for cosine similarity
            device=self.device
        )
        
        # Cache large embedding sets
        if len(texts) > 1000:
            np.save(cache_path, embeddings)
            logger.info(f"Cached embeddings to: {cache_path}")
        
        return embeddings

# ============================================================================
# Optimized Vector Index
# ============================================================================

class OptimizedVectorIndex:
    """FAISS index optimized for Intel Core i7-13700H"""
    
    def __init__(self, dimension: int, index_type: str = 'ivf_hnsw'):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.product_ids = []
        self._create_index()
        
    def _create_index(self):
        """Create optimized FAISS index for your hardware"""
        logger.info(f"Creating {self.index_type} index with dimension {self.dimension}")
        
        if self.index_type == 'flat':
            # Exact search
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
            
        elif self.index_type == 'ivf':
            # IVF with more clusters (more RAM available)
            nlist = 500  # Increased from 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
        elif self.index_type == 'hnsw':
            # HNSW optimized for Intel CPU
            M = 48  # Increased connections for better recall
            self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 128  # Better construction quality
            
        elif self.index_type == 'ivf_hnsw':
            # Composite index for best performance
            nlist = 500
            M = 32
            quantizer = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Use multiple threads for search
        if hasattr(faiss, 'omp_set_num_threads'):
            faiss.omp_set_num_threads(HARDWARE_CONFIG['cpu_cores'])
    
    def add_embeddings(self, embeddings: np.ndarray, product_ids: List[str]):
        """Add embeddings with training for approximate indices"""
        
        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Train if needed
        if self.index_type in ['ivf', 'ivf_hnsw']:
            logger.info(f"Training index with {len(embeddings)} vectors...")
            train_size = min(len(embeddings), 50000)  # Can handle more training data
            train_data = embeddings[:train_size]
            self.index.train(train_data)
        
        # Add in batches for better performance
        batch_size = 10000
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding to index"):
            batch = embeddings[i:i+batch_size]
            self.index.add(batch)
        
        self.product_ids = product_ids
        logger.info(f"Added {len(embeddings)} vectors to index")
    
    def search_optimized(self, query_embeddings: np.ndarray, 
                         k: int = 10, nprobe: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized search with nprobe tuning"""
        
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Set nprobe for IVF indices (search quality vs speed)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        # Set efSearch for HNSW
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = k * 4
        
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

# ============================================================================
# Enhanced Reranker with Batching
# ============================================================================

class OptimizedReranker:
    """Cross-encoder reranking with batch processing"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        # Use L-12 model for better quality with acceptable speed on your hardware
        self.model = CrossEncoder(model_name, max_length=512)
        
    def rerank_batch(self, queries: List[str], 
                     all_candidates: List[List[Tuple[str, str]]], 
                     top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch reranking for multiple queries"""
        
        all_pairs = []
        query_indices = []
        
        # Prepare all pairs
        for i, (query, candidates) in enumerate(zip(queries, all_candidates)):
            for _, text in candidates:
                all_pairs.append([query, text])
                query_indices.append(i)
        
        # Score all pairs at once
        if all_pairs:
            all_scores = self.model.predict(all_pairs, batch_size=32)
        else:
            all_scores = []
        
        # Reorganize results by query
        results = [[] for _ in queries]
        for idx, (query_idx, score) in enumerate(zip(query_indices, all_scores)):
            cand_idx = idx - query_indices.index(query_idx)
            if cand_idx < len(all_candidates[query_idx]):
                product_id = all_candidates[query_idx][cand_idx][0]
                results[query_idx].append((product_id, float(score)))
        
        # Sort and truncate
        final_results = []
        for query_results in results:
            query_results.sort(key=lambda x: x[1], reverse=True)
            final_results.append(query_results[:top_k])
        
        return final_results

# ============================================================================
# Complete Pipeline with Hardware Optimizations
# ============================================================================

class OptimizedSemanticSearchPipeline:
    """Complete pipeline optimized for Dell XPS 15 with 32GB RAM"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"Initializing pipeline with config: {config['name']}")
        
        # Initialize components
        self.data_loader = OptimizedDataLoader()
        self.embedding_manager = EnhancedEmbeddingManager(
            model_name=config['embedding_model'],
            device=config.get('device', 'cpu')
        )
        
        self.vector_index = None
        self.reranker = OptimizedReranker() if config.get('use_reranker') else None
        self.product_texts = {}
        
    def build_and_evaluate(self, df_merged: pd.DataFrame) -> Dict[str, float]:
        """Build index and evaluate in one pass"""
        
        # Split data
        df_train = df_merged[df_merged['split'] == 'train']
        df_test = df_merged[df_merged['split'] == 'test']
        
        # Get unique products
        unique_products = df_merged[['product_id', 'product_title', 'product_description', 
                                    'product_bullet_point', 'product_brand', 'product_color']].drop_duplicates()
        
        # Prepare texts with parallel processing
        logger.info("Preparing product texts...")
        self.product_texts = self.data_loader.prepare_product_texts_parallel(
            unique_products, 
            num_workers=HARDWARE_CONFIG['num_workers']
        )
        
        # Encode products
        logger.info("Encoding products...")
        product_ids = list(self.product_texts.keys())
        texts = list(self.product_texts.values())
        
        embeddings = self.embedding_manager.encode_batch_optimized(
            texts, 
            batch_size=HARDWARE_CONFIG['batch_size']
        )
        
        # Create and populate index
        dimension = embeddings.shape[1]
        self.vector_index = OptimizedVectorIndex(
            dimension, 
            self.config.get('index_type', 'ivf_hnsw')
        )
        self.vector_index.add_embeddings(embeddings, product_ids)
        
        # Prepare evaluation data
        test_queries = df_test.groupby('query_id')['query'].first().to_dict()
        ground_truth = self._prepare_ground_truth(df_test)
        
        # Run evaluation
        logger.info(f"Evaluating on {len(test_queries)} test queries...")
        metrics = self._evaluate_parallel(test_queries, ground_truth)
        
        return metrics
    
    def _prepare_ground_truth(self, df_test: pd.DataFrame) -> Dict:
        """Prepare ground truth labels"""
        ground_truth = {}
        
        for query_id in df_test['query_id'].unique():
            query_data = df_test[df_test['query_id'] == query_id]
            ground_truth[query_id] = {}
            
            for label in ['E', 'S', 'C', 'I']:
                products = query_data[query_data['esci_label'] == label]['product_id'].tolist()
                if products:
                    ground_truth[query_id][label] = products
        
        return ground_truth
    
    def _evaluate_parallel(self, test_queries: Dict, ground_truth: Dict) -> Dict:
        """Parallel evaluation for faster results"""
        
        query_ids = list(test_queries.keys())
        query_texts = list(test_queries.values())
        
        # Process in optimized batches
        batch_size = 256  # Larger batches with more RAM
        all_results = {}
        
        start_time = time.time()
        
        for i in tqdm(range(0, len(query_texts), batch_size), desc="Evaluating"):
            batch_queries = query_texts[i:i+batch_size]
            batch_ids = query_ids[i:i+batch_size]
            
            # Encode queries
            query_embeddings = self.embedding_manager.encode_batch_optimized(
                batch_queries, 
                show_progress=False
            )
            
            # Search
            k = 50 if self.reranker else 10
            distances, indices = self.vector_index.search_optimized(query_embeddings, k=k)
            
            # Convert to product IDs
            batch_results = []
            for query_indices in indices:
                products = [self.vector_index.product_ids[idx] 
                          for idx in query_indices 
                          if idx < len(self.vector_index.product_ids)]
                batch_results.append(products)
            
            # Rerank if enabled
            if self.reranker:
                candidates_batch = []
                for query, products in zip(batch_queries, batch_results):
                    candidates = [(pid, self.product_texts.get(pid, "")) for pid in products]
                    candidates_batch.append(candidates)
                
                reranked = self.reranker.rerank_batch(batch_queries, candidates_batch, top_k=10)
                
                for query_id, reranked_results in zip(batch_ids, reranked):
                    all_results[query_id] = [pid for pid, _ in reranked_results]
            else:
                for query_id, products in zip(batch_ids, batch_results):
                    all_results[query_id] = products[:10]
        
        search_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_results, ground_truth)
        metrics['Total_Time'] = search_time
        metrics['Avg_Query_Time'] = search_time / len(test_queries)
        
        return metrics
    
    def _calculate_metrics(self, results: Dict, ground_truth: Dict) -> Dict:
        """Calculate evaluation metrics"""
        
        all_hits = {1: [], 5: [], 10: []}
        all_mrr = []
        
        for query_id, retrieved in results.items():
            if query_id in ground_truth:
                # Consider Exact and Substitute as relevant
                exact = ground_truth[query_id].get('E', [])
                substitute = ground_truth[query_id].get('S', [])
                relevant = exact + substitute
                
                if relevant:
                    # Hits@K
                    for k in [1, 5, 10]:
                        hit = 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0
                        all_hits[k].append(hit)
                    
                    # MRR
                    for rank, item in enumerate(retrieved[:10], 1):
                        if item in relevant:
                            all_mrr.append(1.0 / rank)
                            break
                    else:
                        all_mrr.append(0.0)
        
        return {
            'Hits@1': np.mean(all_hits[1]),
            'Hits@5': np.mean(all_hits[5]),
            'Hits@10': np.mean(all_hits[10]),
            'MRR': np.mean(all_mrr),
            'Evaluated_Queries': len(all_mrr)
        }

# ============================================================================
# Main Execution with Progress Tracking
# ============================================================================

def main():
    """Main execution optimized for your hardware"""
    
    print("\n" + "="*70)
    print("SEMANTIC SEARCH SOLUTION FOR GRAINGER")
    print("="*70)
    print(f"Hardware: Dell XPS 15 9530")
    print(f"RAM: 32GB")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CPU Cores: {HARDWARE_CONFIG['cpu_cores']}")
    print("="*70 + "\n")
    
    # Define configurations optimized for your hardware
    configurations = [
        {
            'name': 'Baseline (MiniLM-L6)',
            'embedding_model': 'all-MiniLM-L6-v2',
            'use_reranker': False,
            'index_type': 'flat'
        },
        {
            'name': 'MPNet-Base',
            'embedding_model': 'all-mpnet-base-v2',
            'use_reranker': False,
            'index_type': 'ivf_hnsw'  # Better index with more RAM
        },
        {
            'name': 'MPNet + Cross-Encoder',
            'embedding_model': 'all-mpnet-base-v2',
            'use_reranker': True,
            'index_type': 'ivf_hnsw'
        },
        {
            'name': 'MiniLM-L12 (Balanced)',
            'embedding_model': 'all-MiniLM-L12-v2',
            'use_reranker': True,
            'index_type': 'ivf_hnsw'
        }
    ]
    
    # Load data (use large version with 32GB RAM)
    data_loader = OptimizedDataLoader()
    df_merged = data_loader.load_data(use_small_version=False)  # Use large version
    
    # Run experiments
    all_results = []
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")
        
        pipeline = OptimizedSemanticSearchPipeline(config)
        
        start_time = time.time()
        metrics = pipeline.build_and_evaluate(df_merged)
        total_time = time.time() - start_time
        
        metrics['Configuration'] = config['name']
        metrics['Total_Pipeline_Time'] = total_time
        all_results.append(metrics)
        
        # Print results
        print(f"\nResults for {config['name']}:")
        print(f"  Hits@1:  {metrics['Hits@1']:.4f}")
        print(f"  Hits@5:  {metrics['Hits@5']:.4f}")
        print(f"  Hits@10: {metrics['Hits@10']:.4f}")
        print(f"  MRR:     {metrics['MRR']:.4f}")
        print(f"  Avg Query Time: {metrics['Avg_Query_Time']:.3f}s")
        print(f"  Total Time: {total_time:.1f}s")
    
    # Display final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    
    df_results = pd.DataFrame(all_results)
    print(df_results[['Configuration', 'Hits@1', 'Hits@5', 'Hits@10', 'MRR', 'Avg_Query_Time']].to_string(index=False))
    
    # Find best configuration
    best_config = df_results.loc[df_results['MRR'].idxmax()]
    print(f"\n✅ Best Configuration: {best_config['Configuration']}")
    print(f"   MRR: {best_config['MRR']:.4f}")
    print(f"   Hits@10: {best_config['Hits@10']:.4f}")
    
    # Save results
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    
    df_results.to_csv(output_dir / "experiment_results.csv", index=False)
    
    # Generate visualizations
    generate_visualizations(df_results, output_dir)
    
    print(f"\n📊 Results saved to: {output_dir}")
    print("✅ Experiment complete!")
    
    return all_results

def generate_visualizations(df_results: pd.DataFrame, output_dir: Path):
    """Generate performance visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hits@K comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df_results))
    width = 0.25
    ax1.bar(x - width, df_results['Hits@1'], width, label='Hits@1', color='#2E86AB')
    ax1.bar(x, df_results['Hits@5'], width, label='Hits@5', color='#A23B72')
    ax1.bar(x + width, df_results['Hits@10'], width, label='Hits@10', color='#F18F01')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_results['Configuration'], rotation=45, ha='right')
    ax1.set_ylabel('Score')
    ax1.set_title('Hits@K Metrics Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MRR comparison
    ax2 = axes[0, 1]
    ax2.bar(df_results['Configuration'], df_results['MRR'], color='#73AB84')
    ax2.set_xticklabels(df_results['Configuration'], rotation=45, ha='right')
    ax2.set_ylabel('MRR Score')
    ax2.set_title('Mean Reciprocal Rank Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Query time
    ax3 = axes[1, 0]
    ax3.bar(df_results['Configuration'], df_results['Avg_Query_Time'], color='#EE6C4D')
    ax3.set_xticklabels(df_results['Configuration'], rotation=45, ha='right')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Average Query Time')
    ax3.grid(True, alpha=0.3)
    
    # Performance vs Speed tradeoff
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_results['Avg_Query_Time'], df_results['MRR'], 
                         s=200, c=range(len(df_results)), cmap='viridis', alpha=0.7)
    for i, row in df_results.iterrows():
        ax4.annotate(row['Configuration'], 
                    (row['Avg_Query_Time'], row['MRR']),
                    fontsize=9, ha='center', va='bottom')
    ax4.set_xlabel('Average Query Time (seconds)')
    ax4.set_ylabel('MRR Score')
    ax4.set_title('Performance vs Latency Tradeoff')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise