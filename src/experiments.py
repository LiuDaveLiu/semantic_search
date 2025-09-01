from typing import Dict, List, Optional
import pandas as pd

from src.pipeline import SemanticSearchPipeline

def run_optimization_experiments(sample_frac: Optional[float] = None) -> List[Dict]:
    """
    Run optimized experiments based on baseline MRR of 0.4887
    
    Improvements:
    - Reranker: 0.5979
    - Enhanced text + Reranker: 0.6055
    - QA models + Enhanced text + Reranker: 0.5878
    - Deeper models + Enhanced text + Reranker: 0.6050
    """
    
    print("\n" + "="*70)
    print("OPTIMIZATION EXPERIMENTS")
    print("Baseline MRR: 0.4887")
    print("="*70)
    
    results = []
    
    # Experiment 1: Basic text with reranker
    print("\n1. Testing basic text + reranker...")
    config1 = {
        'name': 'MiniLM-L6 + Reranker',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'index_type': 'ivf',
        'use_reranker': True,
        'text_strategy': 'basic'
    }
    pipeline1 = SemanticSearchPipeline(config1)
    metrics1 = pipeline1.run_full_pipeline('small', sample_frac)
    metrics1['name'] = config1['name']
    results.append(metrics1)
    print(f"Result: MRR = {metrics1['MRR']:.4f}")
    
    # Experiment 2: Enhanced text with reranker
    print("\n2. Testing enhanced text + reranker...")
    config2 = {
        'name': 'MiniLM-L6 + Enhanced + Reranker',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'index_type': 'ivf',
        'use_reranker': True,
        'text_strategy': 'enhanced'
    }
    pipeline2 = SemanticSearchPipeline(config2)
    metrics2 = pipeline2.run_full_pipeline('small', sample_frac)
    metrics2['name'] = config2['name']
    results.append(metrics2)
    print(f"Result: MRR = {metrics2['MRR']:.4f}")
    
    # Experiment 3: QA model with enhanced text
    print("\n3. Testing QA-optimized model...")
    config3 = {
        'name': 'QA-MiniLM + Enhanced + Reranker',
        'embedding_model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
        'index_type': 'ivf',
        'use_reranker': True,
        'text_strategy': 'enhanced'
    }
    pipeline3 = SemanticSearchPipeline(config3)
    metrics3 = pipeline3.run_full_pipeline('small', sample_frac)
    metrics3['name'] = config3['name']
    results.append(metrics3)
    print(f"Result: MRR = {metrics3['MRR']:.4f}")
    
    # Experiment 4: Deeper model
    print("\n4. Testing MiniLM-L12 (deeper model)...")
    config4 = {
        'name': 'MiniLM-L12 + Enhanced + Reranker',
        'embedding_model': 'sentence-transformers/all-MiniLM-L12-v2',
        'index_type': 'ivf',
        'use_reranker': True,
        'text_strategy': 'enhanced'
    }
    pipeline4 = SemanticSearchPipeline(config4)
    metrics4 = pipeline4.run_full_pipeline('small', sample_frac)
    metrics4['name'] = config4['name']
    results.append(metrics4)
    print(f"Result: MRR = {metrics4['MRR']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(results)
    print(df[['name', 'Hits@1', 'Hits@5', 'Hits@10', 'MRR']].to_string(index=False))
    
    # Best configuration
    best_idx = df['MRR'].idxmax()
    best = df.iloc[best_idx]
    print(f"\n🏆 Best: {best['name']} with MRR = {best['MRR']:.4f}")
    
    return results
