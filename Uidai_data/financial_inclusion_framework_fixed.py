#!/usr/bin/env python3
"""
Financial Inclusion Scout - UIDAI Data Hackathon
Fixed version with proper column handling
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration"""
    
    VALID_STATES = [
        'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
        'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
        'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
        'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
        'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal',
        'andaman and nicobar islands', 'chandigarh', 
        'dadra and nagar haveli and daman and diu', 'delhi', 
        'jammu and kashmir', 'ladakh', 'lakshadweep', 'puducherry'
    ]
    
    STATE_MAPPING = {
        'orissa': 'odisha', 'uttaranchal': 'uttarakhand', 'pondicherry': 'puducherry',
        'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
        'daman and diu': 'dadra and nagar haveli and daman and diu'
    }
    
    ENROLMENT_WEIGHT = 1.0
    DEMOGRAPHIC_WEIGHT = 0.6
    BIOMETRIC_WEIGHT = 0.8
    
    FORECAST_WEIGHT = 0.4
    FIRS_WEIGHT = 0.3
    GAP_WEIGHT = 0.3


# =============================================================================
# DATA CLEANER
# =============================================================================

class DataCleaner:
    """Handles data cleaning"""
    
    @staticmethod
    def clean_dataset(df, dataset_name):
        """Clean dataset"""
        print(f"\nCleaning {dataset_name}...")
        print(f"Initial: {len(df):,} records")
        
        # Standardize columns
        df.columns = df.columns.str.lower().str.strip()
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df[df['date'] <= pd.Timestamp.now()]
        
        # Clean geographic fields
        df['state'] = df['state'].astype(str).str.lower().str.strip()
        df['state'] = df['state'].replace(Config.STATE_MAPPING)
        df = df[df['state'].isin(Config.VALID_STATES)]
        
        df['district'] = df['district'].astype(str).str.lower().str.strip()
        
        # Create temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'state', 'district'])
        
        print(f"Final: {len(df):,} records")
        return df


# =============================================================================
# MAIN FRAMEWORK
# =============================================================================

class FinancialInclusionScout:
    """Main framework"""
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("FINANCIAL INCLUSION SCOUT - AADHAAR SERVICE LOAD FORECASTING")
        print("="*70)
        
        # Load and clean data
        print("\n[1/4] Loading and cleaning data...")
        enrol = DataCleaner.clean_dataset(
            pd.read_csv('api_data_aadhar_enrolment_0_500000.csv'), 
            "Enrolment"
        )
        demo = DataCleaner.clean_dataset(
            pd.read_csv('api_data_aadhar_demographic_0_500000.csv'), 
            "Demographic"
        )
        bio = DataCleaner.clean_dataset(
            pd.read_csv('api_data_aadhar_biometric_0_500000.csv'), 
            "Biometric"
        )
        
        # Aggregate to monthly
        print("\n[2/4] Aggregating to monthly district level...")
        enrol_monthly = self._aggregate(enrol, 'age_18_greater')
        demo_monthly = self._aggregate(demo, 'age_18_greater')
        bio_monthly = self._aggregate(bio, 'age_18_greater')
        
        # Compute service load
        print("\n[3/4] Computing service load and FIRS...")
        merged = enrol_monthly.merge(demo_monthly, on=['state', 'district', 'year_month'], 
                                     how='outer', suffixes=('_enrol', '_demo'))
        merged = merged.merge(bio_monthly, on=['state', 'district', 'year_month'], how='outer')
        merged = merged.fillna(0)
        
        # Get column names
        cols = merged.columns.tolist()
        enrol_col = [c for c in cols if 'enrol' in c and c not in ['state', 'district', 'year_month']][0]
        demo_col = [c for c in cols if 'demo' in c and c not in ['state', 'district', 'year_month']][0]
        bio_col = [c for c in cols if c not in ['state', 'district', 'year_month', enrol_col, demo_col]][0]
        
        merged['enrolment_load'] = merged[enrol_col]
        merged['demographic_load'] = merged[demo_col]
        merged['biometric_load'] = merged[bio_col]
        merged['total_service_load'] = (
            merged['enrolment_load'] * Config.ENROLMENT_WEIGHT +
            merged['demographic_load'] * Config.DEMOGRAPHIC_WEIGHT +
            merged['biometric_load'] * Config.BIOMETRIC_WEIGHT
        )
        
        # Compute FIRS
        merged['firs'] = (
            0.5 * (merged['enrolment_load'] / (merged['total_service_load'] + 1)) +
            0.3 * (merged['biometric_load'] / (merged['total_service_load'] + 1)) +
            0.2
        )
        
        # Compute trends
        merged = merged.sort_values(['state', 'district', 'year_month'])
        merged['ma_3m'] = merged.groupby(['state', 'district'])['total_service_load'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        merged['ma_6m'] = merged.groupby(['state', 'district'])['total_service_load'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        
        # Generate forecasts
        print("\n[4/4] Generating forecasts and priorities...")
        forecasts = self._forecast(merged)
        priorities = self._prioritize(merged, forecasts)
        
        # Save outputs
        self._save_outputs(merged, forecasts, priorities)
        
        # Display results
        self._display_results(priorities)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        return merged, forecasts, priorities
    
    @staticmethod
    def _aggregate(df, value_col):
        """Aggregate to monthly"""
        return df.groupby(['state', 'district', 'year_month']).agg({
            value_col: 'sum'
        }).reset_index()
    
    @staticmethod
    def _forecast(df):
        """Simple forecast"""
        forecasts = []
        for (state, district), group in df.groupby(['state', 'district']):
            if len(group) < 3:
                continue
            recent_avg = group['total_service_load'].tail(6).mean()
            forecasts.append({
                'state': state,
                'district': district,
                'forecasted_load': recent_avg
            })
        return pd.DataFrame(forecasts)
    
    @staticmethod
    def _prioritize(df, forecasts):
        """Compute priorities"""
        latest = df.groupby(['state', 'district']).tail(1)
        priority = latest.merge(forecasts, on=['state', 'district'], how='left')
        
        priority['norm_forecast'] = (priority['forecasted_load'] - priority['forecasted_load'].min()) / \
                                    (priority['forecasted_load'].max() - priority['forecasted_load'].min() + 1)
        priority['norm_firs'] = 1 - priority['firs']
        priority['service_gap'] = priority['total_service_load'] / (priority['ma_6m'] + 1)
        priority['norm_gap'] = (priority['service_gap'] - priority['service_gap'].min()) / \
                               (priority['service_gap'].max() - priority['service_gap'].min() + 1)
        
        priority['priority_score'] = (
            priority['norm_forecast'] * Config.FORECAST_WEIGHT +
            priority['norm_firs'] * Config.FIRS_WEIGHT +
            priority['norm_gap'] * Config.GAP_WEIGHT
        )
        
        priority = priority.sort_values('priority_score', ascending=False)
        priority['rank'] = range(1, len(priority) + 1)
        
        n = len(priority)
        priority['priority_tier'] = pd.cut(
            priority['rank'],
            bins=[0, n*0.1, n*0.25, n*0.5, n],
            labels=['Critical', 'High', 'Medium', 'Low']
        )
        
        return priority
    
    @staticmethod
    def _save_outputs(service_load, forecasts, priorities):
        """Save outputs"""
        service_load.to_csv('output_service_load.csv', index=False)
        forecasts.to_csv('output_forecasts.csv', index=False)
        priorities.to_csv('output_priority_rankings.csv', index=False)
        priorities.head(20).to_csv('output_camp_recommendations.csv', index=False)
        
        print("\n✓ Saved: output_service_load.csv")
        print("✓ Saved: output_forecasts.csv")
        print("✓ Saved: output_priority_rankings.csv")
        print("✓ Saved: output_camp_recommendations.csv")
    
    @staticmethod
    def _display_results(priorities):
        """Display top results"""
        print("\n" + "-"*70)
        print("TOP 10 PRIORITY DISTRICTS FOR CAMP PLACEMENT")
        print("-"*70)
        top10 = priorities.head(10)[['state', 'district', 'priority_score', 'priority_tier', 
                                      'forecasted_load', 'firs', 'total_service_load']]
        print(top10.to_string(index=False))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    scout = FinancialInclusionScout()
    scout.run()


if __name__ == "__main__":
    main()
