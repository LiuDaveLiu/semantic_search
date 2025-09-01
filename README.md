# semantic_search
This code builds a basic semantic search application and reports the quality of the solution.

required python version is ^3.12

## Results Summary
| Configuration | Hits@1 | Hits@5 | Hits@10 | MRR | Runtime |
|--------------|--------|--------|---------|-----|---------|
| Baseline (MiniLM-L6) | 0.384 | 0.623 | 0.714 | 0.488 | 10 min |
| MiniLM-L6 + Reranker | 0.525 | 0.725 | 0.783 | 0.611 | 35 min |
| MiniLM-L6 + Enhanced Product Text + Reranker | 0.519 | 0.722 | 0.771 | 0.606 | 35 min |
| MiniLM-L12 + Enhanced Product Text + Reranker | 0.519 | 0.721 | 0.772 | 0.605 | 35 min |
| QA-MiniLM + Enhanced Product Text + Reranker | 0.505 | 0.699 | 0.750 | 0.588 | 35 min |

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

## Final Pipeline
Query → Embedding → FAISS Search (top 50) → Cross-Encoder Rerank → Top 10 Results

# Components

Embedding Function

Model: sentence-transformers/all-MiniLM-L6-v2
Dimension: 384
Normalized for cosine similarity


Vector Index

Type: FAISS IVF (Inverted File Index)
Clusters: 256
Metric: Inner Product (cosine similarity)


Reranker

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
Reranks top 50 candidates


## Features
The dataset of products and search queries are derived from Amazon's esci-data dataset. It contains iterations of different approaches to improve search performance
