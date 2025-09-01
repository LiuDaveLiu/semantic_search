import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent))

from src.pipeline import SemanticSearchPipeline
from src.experiments import run_optimization_experiments
from src.utils import save_results

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_single_experiment(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single experiment"""
    
    # Override with CLI arguments
    if args.sample:
        sample_frac = args.sample
    else:
        sample_frac = config.get('data', {}).get('sample_frac')
    
    # Initialize pipeline
    pipeline = SemanticSearchPipeline(config.get('pipeline', config))
    
    # Run
    start_time = time.time()
    
    metrics = pipeline.run_full_pipeline(
        version=config.get('data', {}).get('version', 'small'),
        sample_frac=sample_frac
    )
    
    metrics['runtime_seconds'] = time.time() - start_time
    metrics['config_name'] = config.get('name', 'unnamed')
    
    return metrics

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Semantic Product Search")
    
    parser.add_argument('--config', type=str, default='configs/best.yaml',
                       help='Configuration file')
    parser.add_argument('--sample', type=float, help='Sample fraction for testing')
    parser.add_argument('--optimize', action='store_true', help='Run optimization experiments')
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected - using CPU")
    
    
    # Optimization mode
    if args.optimize:
        print("\n🔬 Running optimization experiments...")
        results = run_optimization_experiments(args.sample)
        df = pd.DataFrame(results)
        save_results(df, Path('results'))
        return
    try:
        # Single experiment
        config = load_config(args.config)
        metrics = run_single_experiment(config, args)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Save
        df = pd.DataFrame([metrics])
        save_results(df, Path('results'))
        
        print("\n✅ Complete! Check results/ for outputs")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()