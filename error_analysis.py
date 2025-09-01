import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

def analyze_search_errors():
    """Analyze where and why search fails"""
    
    print("=" * 60)
    print("ERROR ANALYSIS - Understanding Search Failures")
    print("=" * 60)
    
    # Load index
    index_dir = Path("./indices/basic_all-MiniLM-L6-v2")
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "product_ids.pkl", 'rb') as f:
        product_ids = pickle.load(f)
    with open(index_dir / "product_texts.pkl", 'rb') as f:
        product_texts = pickle.load(f)
    
    # Load models
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Load test data
    df = pd.read_parquet("./data/shopping_queries_dataset_examples.parquet")
    df_products = pd.read_parquet("./data/shopping_queries_dataset_products.parquet")
    
    # Merge to get product info
    df = df.merge(df_products, on=['product_locale', 'product_id'], how='left')
    
    # Filter test set
    test_df = df[(df['split'] == 'test') & (df['small_version'] == 1) & (df['product_locale'] == 'us')]
    
    # Analyze a sample
    test_queries = test_df.groupby('query_id').first().reset_index()
    sample = test_queries.sample(n=min(500, len(test_queries)), random_state=42)
    
    # Categories for analysis
    error_types = defaultdict(list)
    success_examples = []
    failure_examples = []
    
    print(f"\nAnalyzing {len(sample)} queries...")
    
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        query_text = row['query']
        query_id = row['query_id']
        
        # Get ground truth
        query_data = test_df[test_df['query_id'] == query_id]
        exact_matches = set(query_data[query_data['esci_label'] == 'E']['product_id'])
        substitutes = set(query_data[query_data['esci_label'] == 'S']['product_id'])
        relevant = exact_matches | substitutes
        
        if not relevant:
            continue
        
        # Search
        query_embedding = embed_model.encode([query_text], normalize_embeddings=True)
        index.nprobe = 32
        scores, indices = index.search(query_embedding.astype('float32'), 50)
        
        candidates = [product_ids[i] for i in indices[0] if i < len(product_ids)]
        
        # Rerank
        if candidates:
            pairs = [[query_text, product_texts[pid]] for pid in candidates]
            rerank_scores = rerank_model.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            predictions = [pid for pid, _ in ranked[:10]]
        else:
            predictions = candidates[:10]
        
        # Analyze results
        found_exact = any(p in exact_matches for p in predictions[:10])
        found_substitute = any(p in substitutes for p in predictions[:10])
        found_any = found_exact or found_substitute
        
        # Categorize errors
        if not found_any:
            # Complete miss
            error_types['complete_miss'].append({
                'query': query_text,
                'query_length': len(query_text.split()),
                'num_relevant': len(relevant),
                'relevant_sample': list(relevant)[:3]
            })
            
            # Get product info for error analysis
            if exact_matches:
                exact_product = query_data[query_data['product_id'].isin(exact_matches)].iloc[0]
                failure_examples.append({
                    'query': query_text,
                    'expected': exact_product['product_title'],
                    'got': product_texts.get(predictions[0], '')[:100] if predictions else 'None'
                })
        
        elif not found_exact and exact_matches:
            # Found substitute but not exact
            error_types['substitute_only'].append({
                'query': query_text,
                'num_exact': len(exact_matches)
            })
        
        elif found_exact:
            # Success case
            rank = next(i for i, p in enumerate(predictions, 1) if p in exact_matches)
            success_examples.append({
                'query': query_text,
                'rank': rank
            })
            
            if rank > 5:
                error_types['low_rank'].append({
                    'query': query_text,
                    'rank': rank
                })
    
    # Analysis Report
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    total_analyzed = len(success_examples) + len(error_types['complete_miss'])
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total queries analyzed: {total_analyzed}")
    print(f"  Successful searches: {len(success_examples)} ({len(success_examples)/total_analyzed*100:.1f}%)")
    print(f"  Complete failures: {len(error_types['complete_miss'])} ({len(error_types['complete_miss'])/total_analyzed*100:.1f}%)")
    
    print(f"\n❌ Error Breakdown:")
    print(f"  Complete misses: {len(error_types['complete_miss'])}")
    print(f"  Substitute only: {len(error_types['substitute_only'])}")
    print(f"  Low rank (>5): {len(error_types['low_rank'])}")
    
    # Query length analysis
    if error_types['complete_miss']:
        miss_lengths = [e['query_length'] for e in error_types['complete_miss']]
        print(f"\n📏 Query Length Impact on Failures:")
        print(f"  Average length of failed queries: {np.mean(miss_lengths):.1f} words")
        print(f"  1-2 words: {sum(1 for l in miss_lengths if l <= 2)}")
        print(f"  3-4 words: {sum(1 for l in miss_lengths if 3 <= l <= 4)}")
        print(f"  5+ words: {sum(1 for l in miss_lengths if l >= 5)}")
    
    # Example failures
    print(f"\n🔍 Example Failures (Complete Misses):")
    for ex in failure_examples[:20]:
        print(f"\n  Query: '{ex['query']}'")
        print(f"  Expected: {ex['expected']}")
        print(f"  Got: {ex['got']}")
    
    # Success rank distribution
    if success_examples:
        ranks = [e['rank'] for e in success_examples]
        print(f"\n✅ Success Rank Distribution:")
        print(f"  Rank 1: {sum(1 for r in ranks if r == 1)} ({sum(1 for r in ranks if r == 1)/len(ranks)*100:.1f}%)")
        print(f"  Rank 2-3: {sum(1 for r in ranks if 2 <= r <= 3)}")
        print(f"  Rank 4-5: {sum(1 for r in ranks if 4 <= r <= 5)}")
        print(f"  Rank 6-10: {sum(1 for r in ranks if 6 <= r <= 10)}")
    
    # Insights
    print("\n" + "=" * 60)
    print("💡 KEY INSIGHTS")
    print("=" * 60)
    
    insights = []
    
    # Check if short queries fail more
    if error_types['complete_miss']:
        avg_fail_length = np.mean([e['query_length'] for e in error_types['complete_miss']])
        if avg_fail_length < 3:
            insights.append("• Short queries (1-2 words) are more likely to fail")
            insights.append("  → Consider query expansion or adding product categories")
    
    # Check if reranker helps
    if len(error_types['low_rank']) > len(error_types['complete_miss']):
        insights.append("• Many relevant products are found but ranked low")
        insights.append("  → The reranker is crucial for performance")
    
    # Product coverage
    unique_products_in_results = len(set(p for e in success_examples + failure_examples for p in e.get('got', [])))
    insights.append(f"• Search results cover {unique_products_in_results} unique products")
    
    if not insights:
        insights.append("• System is performing well overall")
    
    for insight in insights:
        print(insight)
    
    return error_types, success_examples, failure_examples

if __name__ == "__main__":
    error_types, successes, failures = analyze_search_errors()