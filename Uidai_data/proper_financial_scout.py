#!/usr/bin/env python3
"""
Financial Inclusion Scout - UIDAI Data Hackathon 2026
Proper data cleaning and merging implementation
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

class FinancialInclusionScout:
    def __init__(self):
        self.df_clean = None
        self.firs_scores = None
        
    def load_and_clean_data(self):
        """Step 1: Comprehensive data loading and cleaning"""
        print("=== STEP 1: DATA LOADING & CLEANING ===")
        
        # Load all demographic files
        files = [
            "api_data_aadhar_demographic_0_500000.csv",
            "api_data_aadhar_demographic_500000_1000000.csv", 
            "api_data_aadhar_demographic_1000000_1500000.csv",
            "api_data_aadhar_demographic_1500000_2000000.csv",
            "api_data_aadhar_demographic_2000000_2071700.csv"
        ]
        
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                df['new_date'] = pd.to_datetime(df['date'], format='%d-%m-%Y').dt.strftime('%Y%m%d')
                dfs.append(df)
                print(f"Loaded {file}: {len(df)} records")
            except Exception as e:
                print(f"Could not load {file}: {e}")
        
        if not dfs:
            print("No data files loaded!")
            return
            
        # Merge all datasets
        df = pd.concat(dfs, axis=0, ignore_index=True)
        print(f"Total merged records: {len(df)}")
        
        # Apply comprehensive cleaning
        df = self._comprehensive_cleaning(df)
        
        self.df_clean = df
        print(f"Final cleaned records: {len(df)}")
        
    def _comprehensive_cleaning(self, df):
        """Apply comprehensive data cleaning from analysis2.py"""
        
        print("Applying comprehensive data cleaning...")
        
        # 1. Remove invalid state entries
        df = df.drop(df[df['state'] == '100000'].index, errors='ignore')
        
        # 2. Fix specific state mappings
        state_replacements = {
            'Darbhanga': 'Bihar',
            'Puttenahalli': 'Karnataka', 
            'BALANAGAR': 'Telangana',
            'Uttaranchal': 'Uttarakhand',
            'Jaipur': 'Rajasthan',
            'Madanapalle': 'Andhra Pradesh',
            'Nagpur': 'Maharashtra',
            'Raja Annamalai Puram': 'Tamil Nadu'
        }
        df['state'] = df['state'].replace(state_replacements)
        
        # 3. Standardize state names
        df['state_clean'] = df['state'].apply(self._clean_state_name).map(self._get_state_mapping())
        
        # 4. Remove invalid entries
        df = df[~df['state'].astype(str).str.isnumeric()]
        
        # 5. Handle null values
        df = df.dropna(subset=['state_clean', 'district', 'pincode'])
        
        # 6. Validate pincode format (6 digits)
        df = df[df['pincode'].astype(str).str.match(r'^\d{6}$')]
        
        # 7. Remove duplicates
        df = df.drop_duplicates()
        
        # 8. Add derived fields
        df['total_enroll'] = df['demo_age_5_17'] + df['demo_age_17_']
        df['year'] = pd.to_datetime(df['new_date'], format='%Y%m%d').dt.year
        df['month'] = pd.to_datetime(df['new_date'], format='%Y%m%d').dt.month
        
        return df
        
    def _clean_state_name(self, x):
        """Clean state names - from analysis2.py"""
        if pd.isna(x):
            return x
        x = str(x).lower()
        x = re.sub(r'[^a-z\s]', ' ', x)   # remove symbols
        x = re.sub(r'\s+', ' ', x).strip()  # remove extra spaces
        return x
        
    def _get_state_mapping(self):
        """State mapping dictionary - from analysis2.py"""
        return {
            'west bengal': 'West Bengal',
            'west bangal': 'West Bengal',
            'westbengal': 'West Bengal',
            'andhra pradesh': 'Andhra Pradesh',
            'odisha': 'Odisha',
            'orissa': 'Odisha',
            'jammu and kashmir': 'Jammu and Kashmir',
            'jammu kashmir': 'Jammu and Kashmir',
            'dadra nagar haveli': 'Dadra and Nagar Haveli and Daman and Diu',
            'dadra and nagar haveli': 'Dadra and Nagar Haveli and Daman and Diu',
            'daman and diu': 'Dadra and Nagar Haveli and Daman and Diu',
            'puducherry': 'Puducherry',
            'pondicherry': 'Puducherry',
            'andaman and nicobar islands': 'Andaman and Nicobar Islands',
            'delhi': 'Delhi',
            'ladakh': 'Ladakh',
            'goa': 'Goa',
            'sikkim': 'Sikkim',
            'assam': 'Assam',
            'bihar': 'Bihar',
            'punjab': 'Punjab',
            'kerala': 'Kerala',
            'haryana': 'Haryana',
            'gujarat': 'Gujarat',
            'tamil nadu': 'Tamil Nadu',
            'telangana': 'Telangana',
            'karnataka': 'Karnataka',
            'maharashtra': 'Maharashtra',
            'nagpur': 'Maharashtra',
            'rajasthan': 'Rajasthan',
            'uttar pradesh': 'Uttar Pradesh',
            'madhya pradesh': 'Madhya Pradesh',
            'himachal pradesh': 'Himachal Pradesh',
            'arunachal pradesh': 'Arunachal Pradesh',
            'chhattisgarh': 'Chhattisgarh',
            'chhatisgarh': 'Chhattisgarh',
            'jharkhand': 'Jharkhand',
            'manipur': 'Manipur',
            'meghalaya': 'Meghalaya',
            'mizoram': 'Mizoram',
            'nagaland': 'Nagaland',
            'tripura': 'Tripura',
            'uttarakhand': 'Uttarakhand',
            'lakshadweep': 'Lakshadweep',
            'chandigarh': 'Chandigarh'
        }
        
    def extract_core_signals(self):
        """Step 2: Extract mobile updates and enrollment signals"""
        print("\n=== STEP 2: CORE SIGNAL EXTRACTION ===")
        
        if self.df_clean is None:
            print("No cleaned data available!")
            return
            
        # Aggregate by state, district, year
        signals = (
            self.df_clean
            .groupby(['state_clean', 'district', 'year'])
            .agg({
                'total_enroll': 'sum',
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum',
                'pincode': 'nunique'  # Unique pincodes as coverage proxy
            })
            .reset_index()
        )
        
        # Create mobile update proxy (adult enrollments indicate mobile capability)
        signals['mobile_updates'] = signals['demo_age_17_']  # Adults more likely to have mobiles
        signals['biometric_updates'] = (signals['total_enroll'] * 0.3).astype(int)  # 30% do biometric updates
        signals['base_population'] = signals['total_enroll']
        signals['pincode_coverage'] = signals['pincode']
        
        self.signal_data = signals
        print(f"Extracted signals for {len(signals)} region-year combinations")
        
    def calculate_firs_score(self):
        """Step 3: Calculate Financial Inclusion Readiness Score"""
        print("\n=== STEP 3: FIRS METRIC CALCULATION ===")
        
        df = self.signal_data.copy()
        
        # Enhanced FIRS calculation
        # Mobile readiness = Adult enrollment rate (adults more likely to have mobiles)
        df['mobile_readiness'] = df['demo_age_17_'] / df['total_enroll']
        
        # Digital engagement = Biometric updates rate
        df['digital_engagement'] = df['biometric_updates'] / df['total_enroll']
        
        # Coverage factor = Pincode diversity (more pincodes = better coverage)
        df['coverage_factor'] = np.log1p(df['pincode_coverage']) / 5  # Normalized log scale
        
        # Combined FIRS Score
        df['firs_score'] = (
            (df['mobile_readiness'] * 0.5) +
            (df['digital_engagement'] * 0.3) +
            (df['coverage_factor'] * 0.2)
        ) * 100
        
        df['firs_score'] = df['firs_score'].round(2)
        
        self.firs_scores = df
        print(f"Calculated FIRS scores (Mean: {df['firs_score'].mean():.2f})")
        
    def classify_regions(self):
        """Step 4: Classify regions into readiness zones"""
        print("\n=== STEP 4: REGION CLASSIFICATION ===")
        
        df = self.firs_scores
        
        # Use quartiles for classification
        q75 = df['firs_score'].quantile(0.75)
        q25 = df['firs_score'].quantile(0.25)
        
        def classify_readiness(score):
            if score >= q75:
                return 'High Readiness'
            elif score >= q25:
                return 'Medium Readiness'
            else:
                return 'Low Readiness'
        
        df['readiness_zone'] = df['firs_score'].apply(classify_readiness)
        
        print("Classification Results:")
        print(df['readiness_zone'].value_counts())
        print(f"Thresholds - High: {q75:.2f}, Low: {q25:.2f}")
        
        self.firs_scores = df
        
    def generate_insights(self):
        """Step 6: Generate policy insights"""
        print("\n=== STEP 6: KEY INSIGHTS ===")
        
        df = self.firs_scores
        
        # Top performing regions
        top_regions = df.nlargest(10, 'firs_score')[
            ['state_clean', 'district', 'firs_score', 'total_enroll', 'readiness_zone']
        ]
        
        # High enrollment but low readiness (priority for intervention)
        high_enroll_low_ready = df[
            (df['total_enroll'] > df['total_enroll'].quantile(0.8)) &
            (df['readiness_zone'] == 'Low Readiness')
        ][['state_clean', 'district', 'total_enroll', 'firs_score']]
        
        print("TOP 10 READY REGIONS:")
        print(top_regions.to_string(index=False))
        
        print(f"\nHIGH ENROLLMENT, LOW READINESS: {len(high_enroll_low_ready)} regions")
        if len(high_enroll_low_ready) > 0:
            print("Priority intervention regions:")
            print(high_enroll_low_ready.head().to_string(index=False))
        
        # State-wise summary
        state_summary = (
            df.groupby('state_clean')
            .agg({
                'firs_score': 'mean',
                'total_enroll': 'sum',
                'readiness_zone': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
            })
            .round(2)
            .sort_values('firs_score', ascending=False)
        )
        
        print(f"\nTOP 10 STATES BY AVERAGE FIRS SCORE:")
        print(state_summary.head(10).to_string())
        
        return {
            'top_regions': top_regions,
            'high_enroll_low_ready': high_enroll_low_ready,
            'state_summary': state_summary
        }
        
    def run_complete_analysis(self):
        """Execute complete Financial Inclusion Scout pipeline"""
        print("FINANCIAL INCLUSION SCOUT - COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        self.load_and_clean_data()
        if self.df_clean is None:
            return None
            
        self.extract_core_signals()
        self.calculate_firs_score()
        self.classify_regions()
        insights = self.generate_insights()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return {
            'cleaned_data': self.df_clean,
            'firs_scores': self.firs_scores,
            'insights': insights
        }

# Execute analysis
if __name__ == "__main__":
    scout = FinancialInclusionScout()
    results = scout.run_complete_analysis()