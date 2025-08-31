import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Calculate evaluation metrics for search results"""
    
    @staticmethod
    def calculate_metrics(predictions: Dict[str, List[str]],
                         ground_truth: Dict[str, Dict[str, List[str]]],
                         k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Calculate Hits@K and MRR metrics
        
        Args:
            predictions: Dict of query_id -> list of predicted product_ids
            ground_truth: Dict of query_id -> label -> list of product_ids
            k_values: List of k values for Hits@K
        """
        hits_at_k = {k: [] for k in k_values}
        mrr_scores = []
        ndcg_scores = []
        
        for query_id, predicted in predictions.items():
            if query_id not in ground_truth:
                continue
            
            # Get relevant products (Exact + Substitute)
            exact = ground_truth[query_id].get('E', [])
            substitute = ground_truth[query_id].get('S', [])
            complement = ground_truth[query_id].get('C', [])
            
            relevant = exact + substitute
            
            if not relevant:
                continue
            
            # Hits@K
            for k in k_values:
                if len(predicted) >= k:
                    hit = any(p in relevant for p in predicted[:k])
                    hits_at_k[k].append(1.0 if hit else 0.0)
            
            # MRR
            for rank, product in enumerate(predicted[:10], 1):
                if product in relevant:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)
            
            # NDCG@10
            dcg = 0.0
            for i, product in enumerate(predicted[:10]):
                rel = 0
                if product in exact:
                    rel = 3
                elif product in substitute:
                    rel = 2
                elif product in complement:
                    rel = 1
                
                if i == 0:
                    dcg += rel
                else:
                    dcg += rel / np.log2(i + 1)
            
            # Ideal DCG
            ideal_rels = [3] * len(exact) + [2] * len(substitute) + [1] * len(complement)
            ideal_rels = sorted(ideal_rels, reverse=True)[:10]
            idcg = sum(rel / np.log2(i + 2) if i > 0 else rel 
                      for i, rel in enumerate(ideal_rels))
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        # Calculate averages
        metrics = {}
        for k in k_values:
            metrics[f'Hits@{k}'] = np.mean(hits_at_k[k]) if hits_at_k[k] else 0.0
        
        metrics['MRR'] = np.mean(mrr_scores) if mrr_scores else 0.0
        metrics['NDCG@10'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        metrics['Num_Queries'] = len(mrr_scores)
        
        return metrics
