import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

from src.config import Config

class ESCIDataLoader:
    """Load and preprocess ESCI dataset"""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Config.DATA_DIR
        self.df_examples = None
        self.df_products = None
        self.df_merged = None
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw parquet files"""       
        examples_path = self.data_path / 'shopping_queries_dataset_examples.parquet'
        products_path = self.data_path / 'shopping_queries_dataset_products.parquet'
        
        if not examples_path.exists() or not products_path.exists():
            raise FileNotFoundError(
                f"Dataset files not found in {self.data_path}\n"
            )
        
        self.df_examples = pd.read_parquet(examples_path, engine='pyarrow')
        self.df_products = pd.read_parquet(products_path, engine='pyarrow')
        
        return self.df_examples, self.df_products
    
    def prepare_dataset(self, 
                       version: str = 'small',
                       locale: str = 'us',
                       sample_frac: Optional[float] = None) -> pd.DataFrame:
        """Prepare merged dataset"""
        if self.df_examples is None:
            self.load_raw_data()
        
        df_merged = pd.merge(
            self.df_examples,
            self.df_products,
            on=['product_locale', 'product_id'],
            how='left'
        )
        
        # Filter version and locale
        version_col = f'{version}_version'
        df_merged = df_merged[df_merged[version_col] == 1]
        df_merged = df_merged[df_merged['product_locale'] == locale]
        
        # Sample if requested
        if sample_frac:
            df_merged = df_merged.sample(frac=sample_frac, random_state=42)
        
        self.df_merged = df_merged
        return df_merged
    
    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train and test splits"""
        if self.df_merged is None:
            raise ValueError("Call prepare_dataset first")
        
        df_train = self.df_merged[self.df_merged['split'] == 'train']
        df_test = self.df_merged[self.df_merged['split'] == 'test']
        
        return df_train, df_test
    
    def get_unique_products(self) -> pd.DataFrame:
        """Get unique products"""
        if self.df_merged is None:
            raise ValueError("Call prepare_dataset first")
        
        return self.df_merged[[
            'product_id', 'product_title', 'product_description',
            'product_bullet_point', 'product_brand', 'product_color'
        ]].drop_duplicates('product_id').reset_index(drop=True)
    
    def prepare_product_texts(self, 
                             df_products: pd.DataFrame,
                             strategy: str = 'enhanced') -> Dict[str, str]:
        """
        Prepare product texts with different strategies
        
        Args:
            df_products: Product dataframe
            strategy: 'basic' or 'enhanced'
        """
        if strategy == 'enhanced':
            return self._prepare_enhanced_texts(df_products)
        else:
            return self._prepare_basic_texts(df_products)
    
    def _prepare_basic_texts(self, df_products: pd.DataFrame) -> Dict[str, str]:
        """Basic text preparation"""
        product_texts = {}
        
        for _, row in tqdm(df_products.iterrows(), total=len(df_products), desc="Processing (basic)"):
            text_parts = []
            
            if pd.notna(row.get('product_title')):
                text_parts.append(str(row['product_title']))
            
            if pd.notna(row.get('product_brand')):
                brand = str(row['product_brand'])
                if brand.lower() not in str(row.get('product_title', '')).lower():
                    text_parts.append(f"Brand: {brand}")
            
            if pd.notna(row.get('product_description')):
                desc = str(row['product_description'])[:Config.MAX_DESCRIPTION_LENGTH]
                text_parts.append(desc)
            
            product_texts[row['product_id']] = " ".join(text_parts)
        
        return product_texts
    
    def _prepare_enhanced_texts(self, df_products: pd.DataFrame) -> Dict[str, str]:
        """
        Enhanced text preparation based on baseline results
        Key improvements:
        1. Repeat important fields
        2. Add brand at beginning
        3. Include more bullet points
        4. Better field ordering
        """
        product_texts = {}
        
        for _, row in tqdm(df_products.iterrows(), total=len(df_products), desc="Processing (enhanced)"):
            text_parts = []
            
            # 1. Brand at beginning (if exists and not in title)
            brand = str(row.get('product_brand', '')).strip()
            title = str(row.get('product_title', '')).strip()
            
            if brand and brand != 'nan' and brand.lower() not in title.lower():
                text_parts.append(brand)
            
            # 2. Title (most important - repeat for emphasis)
            if title and title != 'nan':
                text_parts.append(title)
                if Config.MAX_TITLE_REPEATS > 1:
                    text_parts.append(f"Product: {title}")
            
            # 3. Color (important for exact matches)
            if Config.INCLUDE_COLOR:
                color = str(row.get('product_color', '')).strip()
                if color and color != 'nan' and color.lower() not in title.lower():
                    text_parts.append(f"Color: {color}")
            
            # 4. Brand again with label
            if Config.INCLUDE_BRAND_PREFIX and brand and brand != 'nan':
                text_parts.append(f"Brand: {brand}")
            
            # 5. Bullet points (contain key features)
            bullets = str(row.get('product_bullet_point', '')).strip()
            if bullets and bullets != 'nan':
                bullets = bullets[:Config.MAX_BULLETS_LENGTH]
                text_parts.append(bullets)
            
            # 6. Description (more context)
            desc = str(row.get('product_description', '')).strip()
            if desc and desc != 'nan':
                desc = desc[:Config.MAX_DESCRIPTION_LENGTH]
                text_parts.append(desc)
            
            # Join and clean
            product_text = " ".join(text_parts)
            product_text = " ".join(product_text.split())  # Clean multiple spaces
            
            product_texts[row['product_id']] = product_text
        
        return product_texts
    
    def get_ground_truth(self, df_test: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Extract ground truth from test set"""
        ground_truth = {}
        
        for query_id in df_test['query_id'].unique():
            query_data = df_test[df_test['query_id'] == query_id]
            ground_truth[query_id] = {}
            
            for label in ['E', 'S', 'C', 'I']:
                products = query_data[query_data['esci_label'] == label]['product_id'].tolist()
                if products:
                    ground_truth[query_id][label] = products
        
        return ground_truth
    
    def get_test_queries(self, df_test: pd.DataFrame) -> Dict[str, str]:
        """Get test queries"""
        return df_test.groupby('query_id')['query'].first().to_dict()