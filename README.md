# semantic_search
This code builds a basic semantic search application and reports the quality of the solution.

required python version is ^3.12

## Run

### Install dependency
```shell
> python -m venv venv
> venv\Scripts\activate
> pip install -r requirements.txt
```

### Download Data
https://github.com/amazon-science/esci-data

Download these files to semantic_search/data/
shopping_queries_dataset_examples.parquet
shopping_queries_dataset_products.parquet
shopping_queries_dataset_sources.csv

### Run
```shell
> python src/main.py
```
# Or use the notebook for interactive exploration
./notebooks/exploration_and_analysis.ipynb


## Features
The dataset of products and search queries are derived from Amazon's esci-data dataset. It contains iterations of different approaches to improve search performance

