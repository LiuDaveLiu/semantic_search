# semantic_search
This code builds a basic semantic search application and reports the quality of the solution.

required python version is ^3.12

## Results Summary
| Configuration | Hits@1 | Hits@5 | Hits@10 | MRR | Runtime |
|--------------|--------|--------|---------|-----|---------|
| Baseline (MiniLM-L6) | 0.384 | 0.631 | 0.713 | 0.489 | 10 min |
| MiniLM-L6 + Reranker | 0.513 | 0.711 | 0.767 | 0.598 | 35 min |
| MiniLM-L6 + Enhanced Product Text + Reranker | 0.519 | 0.722 | 0.771 | 0.606 | 35 min |
| MiniLM-L12 + Enhanced Product Text + Reranker | 0.519 | 0.721 | 0.772 | 0.605 | 35 min |

### Install dependency
```shell
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers faiss-cpu pandas numpy tqdm pyyaml
```

### Download Data
https://github.com/amazon-science/esci-data

Download these files to semantic_search/data/
shopping_queries_dataset_examples.parquet
shopping_queries_dataset_products.parquet
shopping_queries_dataset_sources.csv

### Run
```shell
# Verify baseline 
python main.py --config configs/baseline.yaml

# Run optimizations
python main.py --optimize

# Run best configuration
python main.py --config configs/best.yaml
```

# Or use the notebook for interactive exploration
./notebooks/experiment.ipynb

## Features
The dataset of products and search queries are derived from Amazon's esci-data dataset. It contains iterations of different approaches to improve search performance
