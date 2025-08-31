import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def setup_logging(verbose: bool = False):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('semantic_search.log'),
            logging.StreamHandler()
        ]
    )

def print_banner():
    """Print banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          SEMANTIC PRODUCT SEARCH - GRAINGER ML              ║
╚══════════════════════════════════════════════════════════════╝
    """)

def save_results(df: pd.DataFrame, output_dir: Path):
    """Save results"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(output_dir / f'results_{timestamp}.csv', index=False)
    
    # Generate plot
    if len(df) > 1:
        fig, axes = plt.subplots(1, 4, figsize=(12, 5))
        
        ax2 = axes[0]
        ax2.bar(range(len(df)), df['Hits@1'])
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Hits@1')
        ax2.set_title('Hits@1 Comparison')

        ax3 = axes[1]
        ax3.bar(range(len(df)), df['Hits@5'])
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Hits@5')
        ax3.set_title('Hits@5 Comparison')

        ax4 = axes[2]
        ax4.bar(range(len(df)), df['Hits@10'])
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Hits@10')
        ax4.set_title('Hits@10 Comparison')

        # MRR comparison
        ax1 = axes[3]
        ax1.bar(range(len(df)), df['MRR'])
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('MRR')
        ax1.set_title('MRR Comparison')

        plt.tight_layout()
        plt.savefig(output_dir / 'results.png')
    
    logging.info(f"Results saved to {output_dir}")