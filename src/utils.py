from pathlib import Path
import pandas as pd
from datetime import datetime

def save_results(df: pd.DataFrame, output_dir: Path):
    """Save results"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(output_dir / f'results_{timestamp}.csv', index=False)