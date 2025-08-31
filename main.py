import argparse
import logging
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
from src.utils import setup_logging, print_banner, save_results

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
    logging.info(f"Running: {config.get('name', 'unnamed')}")
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
    parser.add_argument('--test', action='store_true', help='Quick test with 10% sample')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--optimize', action='store_true', help='Run optimization experiments')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(verbose=args.verbose)
    print_banner()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected - using CPU")
    
    # Test mode
    if args.test:
        print("\n🧪 Running TEST mode (10% sample)")
        args.sample = 0.1
        args.config = 'configs/baseline.yaml'
    
    # Optimization mode
    if args.optimize:
        print("\n🔬 Running optimization experiments...")
        results = run_optimization_experiments(args.sample)
        df = pd.DataFrame(results)
        save_results(df, Path('results'))
        return
    
    try:
        # Run experiments
        if args.all:
            config = load_config(args.config)
            results = []
            
            for exp_config in config.get('experiments', []):
                try:
                    metrics = run_single_experiment(exp_config, args)
                    results.append(metrics)
                    
                    print(f"\n✅ {exp_config['name']}:")
                    print(f"   MRR: {metrics['MRR']:.4f}")
                    print(f"   Hits@10: {metrics['Hits@10']:.4f}")
                    
                except Exception as e:
                    logging.error(f"Failed: {e}")
                    continue
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save results
            if results:
                df = pd.DataFrame(results)
                save_results(df, Path('results'))
                
                print("\n" + "="*60)
                print("SUMMARY")
                print("="*60)
                print(df[['config_name', 'MRR', 'Hits@10']].to_string(index=False))
        
        else:
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
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()