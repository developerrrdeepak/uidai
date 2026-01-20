"""
Data Processing Module for Financial Inclusion Risk Scoring Model
Handles data loading, preprocessing, and indicator calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process UIDAI district-level data and calculate risk indicators"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data processor
        
        Args:
            data_path: Path to the district data CSV/JSON file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self._required_columns = {
            'state', 'total_enrolments', 'male_enrolments', 'female_enrolments',
            'child_0_5_enrolments', 'biometric_updates', 'current_month_enrolments',
            'previous_month_enrolments'
        }
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load district-level UIDAI data
        
        Args:
            data_path: Path to data file (CSV or JSON)
            
        Returns:
            DataFrame with raw district data
            
        Raises:
            ValueError: If file format is unsupported or path is invalid
            FileNotFoundError: If file doesn't exist
        """
        path = data_path or self.data_path
        
        if not path:
            raise ValueError("No data path provided")
        
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            if path.endswith('.csv'):
                self.raw_data = pd.read_csv(path)
            elif path.endswith('.json'):
                self.raw_data = pd.read_json(path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            self._validate_data(self.raw_data)
            logger.info(f"Successfully loaded {len(self.raw_data)} records from {path}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns"""
        missing_cols = self._required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
    
    def calculate_enrolment_coverage(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Enrolment Coverage Proxy
        
        Coverage = district_enrolments / state_average_enrolments
        Low value → access problem
        
        Args:
            df: DataFrame with 'total_enrolments' and 'state' columns
            
        Returns:
            Series with coverage values
        """
        state_avg = df.groupby('state')['total_enrolments'].transform('mean')
        coverage = df['total_enrolments'] / state_avg
        return coverage.replace([np.inf, -np.inf], 0).fillna(0)
    
    def calculate_child_inclusion_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Child Inclusion Ratio (0-5 years)
        
        Child_Ratio = child_0_5_enrolments / total_enrolments
        Low value → long-term risk
        
        Args:
            df: DataFrame with 'child_0_5_enrolments' and 'total_enrolments'
            
        Returns:
            Series with child inclusion ratios
        """
        child_ratio = df['child_0_5_enrolments'] / df['total_enrolments']
        return child_ratio.replace([np.inf, -np.inf], 0).fillna(0)
    
    def calculate_gender_gap_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Gender Gap Index
        
        Gender_Gap = |male - female| / total
        High value → social barrier
        
        Args:
            df: DataFrame with 'male_enrolments', 'female_enrolments', 'total_enrolments'
            
        Returns:
            Series with gender gap values
        """
        gender_gap = abs(df['male_enrolments'] - df['female_enrolments']) / df['total_enrolments']
        return gender_gap.replace([np.inf, -np.inf], 0).fillna(0)
    
    def calculate_update_readiness_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Update Readiness Ratio
        
        Update_Ratio = biometric_updates / total_enrolments
        Low value → DBT failure risk
        
        Args:
            df: DataFrame with 'biometric_updates' and 'total_enrolments'
            
        Returns:
            Series with update readiness ratios
        """
        update_ratio = df['biometric_updates'] / df['total_enrolments']
        return update_ratio.replace([np.inf, -np.inf], 0).fillna(0)
    
    def calculate_enrolment_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Enrolment Momentum
        
        Momentum = (current_month - previous_month) / previous_month
        Negative → exclusion trend
        
        Args:
            df: DataFrame with 'current_month_enrolments' and 'previous_month_enrolments'
            
        Returns:
            Series with momentum values
        """
        momentum = (df['current_month_enrolments'] - df['previous_month_enrolments']) / df['previous_month_enrolments']
        return momentum.replace([np.inf, -np.inf], 0).fillna(0)
    
    def normalize_minmax(self, series: pd.Series) -> pd.Series:
        """
        Normalize series using Min-Max scaling to [0, 1]
        
        X_norm = (X - min) / (max - min)
        
        Args:
            series: Input series to normalize
            
        Returns:
            Normalized series
        """
        min_val = series.min()
        max_val = series.max()
        
        if max_val - min_val == 0:
            return pd.Series(0, index=series.index)
        
        normalized = (series - min_val) / (max_val - min_val)
        return normalized.replace([np.inf, -np.inf], 0).fillna(0)
    
    def align_risk_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align all indicators to risk direction (higher = worse)
        
        Transformations:
        - Coverage: 1 - Coverage (low coverage = high risk)
        - Child Ratio: 1 - Child_Ratio (low ratio = high risk)
        - Gender Gap: Gender_Gap (high gap = high risk)
        - Update Ratio: 1 - Update_Ratio (low updates = high risk)
        - Momentum: max(0, -Momentum) (negative momentum = high risk)
        
        Args:
            df: DataFrame with normalized indicators
            
        Returns:
            DataFrame with risk-aligned indicators
        """
        risk_df = df.copy()
        
        # Invert coverage (low coverage = high risk)
        risk_df['coverage_risk'] = 1 - risk_df['coverage_norm']
        
        # Invert child ratio (low ratio = high risk)
        risk_df['child_risk'] = 1 - risk_df['child_ratio_norm']
        
        # Gender gap already aligned (high gap = high risk)
        risk_df['gender_risk'] = risk_df['gender_gap_norm']
        
        # Invert update ratio (low updates = high risk)
        risk_df['update_risk'] = 1 - risk_df['update_ratio_norm']
        
        risk_df['momentum_risk'] = risk_df['momentum_norm'].apply(lambda x: max(0, -x) if x < 0 else 0)
        
        return risk_df
    
    def process_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 5 indicators, normalize, and align risk direction
        
        Args:
            df: Raw district data DataFrame
            
        Returns:
            DataFrame with all processed indicators
        """
        result_df = df.copy()
        
        # Calculate raw indicators
        result_df['coverage'] = self.calculate_enrolment_coverage(df)
        result_df['child_ratio'] = self.calculate_child_inclusion_ratio(df)
        result_df['gender_gap'] = self.calculate_gender_gap_index(df)
        result_df['update_ratio'] = self.calculate_update_readiness_ratio(df)
        result_df['momentum'] = self.calculate_enrolment_momentum(df)
        
        # Normalize indicators
        result_df['coverage_norm'] = self.normalize_minmax(result_df['coverage'])
        result_df['child_ratio_norm'] = self.normalize_minmax(result_df['child_ratio'])
        result_df['gender_gap_norm'] = self.normalize_minmax(result_df['gender_gap'])
        result_df['update_ratio_norm'] = self.normalize_minmax(result_df['update_ratio'])
        result_df['momentum_norm'] = self.normalize_minmax(result_df['momentum'])
        
        # Align risk direction
        result_df = self.align_risk_direction(result_df)
        
        self.processed_data = result_df
        
        return result_df
    
    def generate_sample_data(self, num_districts: int = 50) -> pd.DataFrame:
        """
        Generate sample UIDAI district data for demonstration
        
        Args:
            num_districts: Number of districts to generate
            
        Returns:
            DataFrame with sample district data
        """
        np.random.seed(42)
        
        states = ['Maharashtra', 'Uttar Pradesh', 'Bihar', 'West Bengal', 'Madhya Pradesh', 
                  'Tamil Nadu', 'Rajasthan', 'Karnataka', 'Gujarat', 'Andhra Pradesh']
        
        districts = [f"District_{i+1}" for i in range(num_districts)]
        
        data = {
            'district': districts,
            'state': np.random.choice(states, num_districts),
            'total_enrolments': np.random.randint(100000, 1000000, num_districts),
            'male_enrolments': np.random.randint(50000, 500000, num_districts),
            'female_enrolments': np.random.randint(50000, 500000, num_districts),
            'child_0_5_enrolments': np.random.randint(5000, 100000, num_districts),
            'biometric_updates': np.random.randint(10000, 500000, num_districts),
            'current_month_enrolments': np.random.randint(1000, 50000, num_districts),
            'previous_month_enrolments': np.random.randint(1000, 50000, num_districts)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure male + female approximately equals total
        df['total_enrolments'] = df['male_enrolments'] + df['female_enrolments']
        
        return df
    
    def save_processed_data(self, output_path: str, file_format: str = 'json') -> None:
        """
        Save processed data to file
        
        Args:
            output_path: Path to save the file
            file_format: 'json' or 'csv'
            
        Raises:
            ValueError: If no processed data or invalid format
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_all_indicators first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_format == 'json':
                self.processed_data.to_json(output_path, orient='records', indent=2)
            elif file_format == 'csv':
                self.processed_data.to_csv(output_path, index=False)
            else:
                raise ValueError("Format must be 'json' or 'csv'")
            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise


if __name__ == "__main__":
    try:
        processor = DataProcessor()
        
        logger.info("Generating sample district data...")
        sample_data = processor.generate_sample_data(num_districts=50)
        
        logger.info("Processing indicators...")
        processed_data = processor.process_all_indicators(sample_data)
        
        print("\nProcessed Data (first 5 districts):")
        print(processed_data[['district', 'state', 'coverage_risk', 'child_risk', 
                              'gender_risk', 'update_risk', 'momentum_risk']].head())
        
        logger.info("Saving processed data...")
        processor.save_processed_data('processed_district_data.json', file_format='json')
        logger.info("Processing completed successfully!")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise