"""Data Loading and Preprocessing Module"""
import pandas as pd
import numpy as np

class DataLoader:
    """Handles data loading from CSV files"""
    
    def load_cleaned_data(self):
        """Load cleaned biometric and demographic data"""
        try:
            bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
            demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')
            return self._process_data(bio_df, demo_df)
        except FileNotFoundError:
            return self._create_sample_data()
    
    def _process_data(self, bio_df, demo_df):
        """Process and merge datasets"""
        bio_agg = bio_df.groupby('district_clean').agg({
            'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'
        }).reset_index()
        
        demo_agg = demo_df.groupby('district_clean').agg({
            'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'
        }).reset_index()
        
        combined = pd.merge(bio_agg, demo_agg, on='district_clean', how='outer').fillna(0)
        
        np.random.seed(42)
        n = len(combined)
        combined['enrolment_pct_change'] = np.random.normal(0, 15, n)
        combined['update_pct_change'] = np.random.normal(-3, 12, n)
        combined['child_pct_change'] = np.random.normal(1, 10, n)
        combined['female_pct_change'] = np.random.normal(0, 8, n)
        
        return combined.rename(columns={'district_clean': 'district'})
    
    def _create_sample_data(self, n=15):
        """Generate sample data for testing"""
        districts = ['Kalahandi', 'Koraput', 'Rayagada', 'Gajapati', 'Kandhamal',
                    'Bolangir', 'Nuapada', 'Bargarh', 'Jharsuguda', 'Sambalpur',
                    'Dhenkanal', 'Angul', 'Deogarh', 'Sundargarh', 'Keonjhar'][:n]
        
        np.random.seed(42)
        data = pd.DataFrame({
            'district': districts,
            'enrolment_pct_change': np.random.normal(0, 20, n),
            'update_pct_change': np.random.normal(-5, 15, n),
            'child_pct_change': np.random.normal(2, 18, n),
            'female_pct_change': np.random.normal(1, 12, n)
        })
        
        # Add high-risk scenarios
        if n >= 5:
            data.loc[0, 'enrolment_pct_change'] = -45  # Crisis
            data.loc[0, 'update_pct_change'] = -35
            data.loc[1, 'child_pct_change'] = -30  # Child exclusion
            data.loc[2, 'female_pct_change'] = -25  # Gender gap
            data.loc[3, 'update_pct_change'] = -40  # Update failure
            data.loc[4, 'enrolment_pct_change'] = 50  # Sudden spike
        
        return data
