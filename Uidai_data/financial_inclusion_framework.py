"""
Financial Inclusion Scout: Aadhaar Service Load Forecasting Framework
UIDAI Data Hackathon Submission

Author: Senior Data Analyst & Public Policy Data Scientist
Version: 1.0

PROBLEM DEFINITION:

India's ambitious financial inclusion agenda faces a critical infrastructure challenge:
uneven distribution of Aadhaar service availability across geographical regions. This
disparity creates cascading barriers to essential welfare and financial services:

- DBT Delivery Gaps: Beneficiaries in underserved regions face difficulties updating
  demographic information or renewing biometric authentication, leading to benefit
  exclusion and payment failures.

- KYC Readiness Constraints: Limited access to Aadhaar update services impedes
  financial account opening, mobile SIM activation, and digital payment adoption,
  particularly affecting rural and remote populations.

- Service Continuity Risks: Aging biometric data and outdated demographic information
  compromise authentication success rates, threatening the integrity of welfare
  delivery systems.

The absence of predictive frameworks for Aadhaar service demand results in reactive
resource allocation, inefficient camp placement, and persistent inclusion gaps that
disproportionately affect vulnerable populations.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for all parameters"""
    
    # Valid Indian states and union territories
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
    
    # State name mappings for standardization
    STATE_MAPPING = {
        'orissa': 'odisha',
        'uttaranchal': 'uttarakhand',
        'pondicherry': 'puducherry',
        'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
        'daman and diu': 'dadra and nagar haveli and daman and diu'
    }
    
    # Service load weights
    ENROLMENT_WEIGHT = 1.0
    DEMOGRAPHIC_WEIGHT = 0.6
    BIOMETRIC_WEIGHT = 0.8
    
    # FIRS component weights
    ADULT_ENROL_WEIGHT = 0.5
    BIO_UPDATE_WEIGHT = 0.3
    PINCODE_COVERAGE_WEIGHT = 0.2
    
    # Priority score weights
    FORECAST_WEIGHT = 0.4
    FIRS_WEIGHT = 0.3
    GAP_WEIGHT = 0.3
    
    # Forecasting parameters
    LOOKBACK_WINDOW = 6
    SMOOTHING_ALPHA = 0.3
    FORECAST_HORIZON = 6
    
    # Priority tier thresholds
    CRITICAL_THRESHOLD = 0.10
    HIGH_THRESHOLD = 0.25
    MEDIUM_THRESHOLD = 0.50
    
    # Output parameters
    TOP_N_RECOMMENDATIONS = 20


# =============================================================================
# MODULE 1: DATA CLEANING
# =============================================================================

class DataCleaner:
    """Handles data validation, standardization, and cleaning"""
    
    def __init__(self):
        self.valid_states = Config.VALID_STATES
        self.state_mapping = Config.STATE_MAPPING
    
    def clean_dataset(self, df, dataset_name):
        """Main cleaning pipeline"""
        print(f"\n{'='*60}")
        print(f"Cleaning {dataset_name}")
        print(f"{'='*60}")
        initial_count = len(df)
        print(f"Initial records: {initial_count:,}")
        
        df = self._handle_missing_values(df)
        df = self._standardize_geography(df)
        df = self._clean_dates(df)
        df = self._remove_duplicates(df)
        df = self._validate_numeric(df)
        
        final_count = len(df)
        print(f"Final records: {final_count:,}")
        print(f"Records removed: {initial_count - final_count:,} ({100*(initial_count-final_count)/initial_count:.1f}%)")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle null and missing values"""
        critical_cols = ['state', 'district', 'date']
        existing_cols = [col for col in critical_cols if col in df.columns]

        # Drop records with missing critical fields
        df = df.dropna(subset=existing_cols)

        # Handle zero and invalid values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace zeros with NaN for logical handling
            df[col] = df[col].replace(0, np.nan)
            # Fill with median for each state-district group
            df[col] = df.groupby(['state', 'district'])[col].transform(
                lambda x: x.fillna(x.median()) if not x.dropna().empty else x.fillna(0)
            )

        return df
    
    def _standardize_geography(self, df):
        """Standardize state, district, PIN code fields"""
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.lower().str.strip()
            df['state'] = df['state'].replace(self.state_mapping)
            df = df[df['state'].isin(self.valid_states)]
        
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.lower().str.strip()
        
        if 'pincode' in df.columns:
            df['pincode'] = df['pincode'].astype(str).str.strip()
            df['pincode'] = df['pincode'].apply(
                lambda x: x if x.isdigit() and len(x) == 6 else np.nan
            )
        
        return df
    
    def _clean_dates(self, df):
        """Parse and validate date fields"""
        if 'date' not in df.columns:
            return df
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df[df['date'] <= pd.Timestamp.now()]
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M')
        df['new_date'] = df['date'].dt.strftime('%Y%m%d')
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate records"""
        key_cols = ['date', 'state', 'district']
        if 'pincode' in df.columns:
            key_cols.append('pincode')
        existing_cols = [col for col in key_cols if col in df.columns]
        return df.drop_duplicates(subset=existing_cols)
    
    def _validate_numeric(self, df):
        """Validate numeric fields"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = df[col].clip(lower=0)
        return df


# =============================================================================
# MODULE 2: DATA PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """Handles aggregation and preprocessing"""
    
    @staticmethod
    def aggregate_to_monthly(df, value_col):
        """Aggregate daily data to monthly district level"""
        agg_dict = {value_col: 'sum'}
        monthly = df.groupby(['state', 'district', 'year_month']).agg(agg_dict).reset_index()
        monthly['year_month'] = monthly['year_month'].astype(str)
        return monthly
    
    @staticmethod
    def aggregate_pincode_level(df, value_col):
        """Aggregate at PIN code level"""
        if 'pincode' not in df.columns:
            return None
        
        agg_dict = {value_col: 'sum'}
        pincode = df.groupby(['state', 'district', 'pincode', 'year_month']).agg(agg_dict).reset_index()
        pincode['year_month'] = pincode['year_month'].astype(str)
        return pincode


# =============================================================================
# MODULE 3: FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Constructs derived features and metrics"""
    
    def __init__(self):
        self.enrol_weight = Config.ENROLMENT_WEIGHT
        self.demo_weight = Config.DEMOGRAPHIC_WEIGHT
        self.bio_weight = Config.BIOMETRIC_WEIGHT
    
    def compute_service_load(self, enrolment_df, demographic_df, biometric_df):
        """Compute total service load with weighted components"""
        # Merge datasets
        merged = enrolment_df.merge(
            demographic_df, 
            on=['state', 'district', 'year_month'], 
            how='outer', 
            suffixes=('_enrol', '_demo')
        )
        merged = merged.merge(
            biometric_df, 
            on=['state', 'district', 'year_month'], 
            how='outer'
        )
        merged = merged.fillna(0)
        
        # Identify value columns
        enrol_col = [c for c in merged.columns if 'enrol' in c.lower() and c not in ['state', 'district', 'year_month']][0]
        demo_col = [c for c in merged.columns if 'demo' in c.lower() and c not in ['state', 'district', 'year_month']][0]
        bio_col = [c for c in merged.columns if c not in ['state', 'district', 'year_month', enrol_col, demo_col]][0]
        
        # Compute weighted service load
        merged['enrolment_load'] = merged[enrol_col]
        merged['demographic_load'] = merged[demo_col]
        merged['biometric_load'] = merged[bio_col]
        merged['total_service_load'] = (
            merged['enrolment_load'] * self.enrol_weight +
            merged['demographic_load'] * self.demo_weight +
            merged['biometric_load'] * self.bio_weight
        )
        
        return merged[['state', 'district', 'year_month', 'enrolment_load', 
                      'demographic_load', 'biometric_load', 'total_service_load']]
    
    def compute_inclusion_readiness(self, df):
        """Compute Financial Inclusion Readiness Score (FIRS)"""
        df['adult_enrol_rate'] = 0.75  # Default proxy
        df['bio_update_rate'] = df['biometric_load'] / (df['enrolment_load'] + 1)
        df['pincode_coverage'] = 1.0  # Default
        
        df['firs'] = (
            df['adult_enrol_rate'] * Config.ADULT_ENROL_WEIGHT +
            df['bio_update_rate'].clip(0, 1) * Config.BIO_UPDATE_WEIGHT +
            df['pincode_coverage'] * Config.PINCODE_COVERAGE_WEIGHT
        )
        
        return df
    
    def compute_trends(self, df):
        """Compute growth rates and moving averages"""
        df = df.sort_values(['state', 'district', 'year_month'])
        
        # Year-over-year growth
        df['yoy_growth'] = df.groupby(['state', 'district'])['total_service_load'].pct_change(12)
        
        # Moving averages
        df['ma_3m'] = df.groupby(['state', 'district'])['total_service_load'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df['ma_6m'] = df.groupby(['state', 'district'])['total_service_load'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        
        return df


# =============================================================================
# MODULE 4: FORECASTING
# =============================================================================

class ServiceLoadForecaster:
    """Simple interpretable forecasting methods"""
    
    def __init__(self):
        self.lookback = Config.LOOKBACK_WINDOW
        self.alpha = Config.SMOOTHING_ALPHA
    
    def forecast_moving_average(self, df, horizon=None):
        """3-6 month forecast using moving average with growth adjustment"""
        if horizon is None:
            horizon = Config.FORECAST_HORIZON

        df = df.sort_values(['state', 'district', 'year_month'])
        forecasts = []

        for (state, district), group in df.groupby(['state', 'district']):
            if len(group) < 6:  # Need at least 6 months for reliable forecast
                continue

            # Calculate 6-month moving average
            recent_avg = group['total_service_load'].tail(self.lookback).mean()

            # Calculate year-over-year growth rate
            if len(group) >= 12:
                current_year_avg = group['total_service_load'].tail(6).mean()
                prev_year_avg = group['total_service_load'].iloc[-12:-6].mean()
                growth_rate = (current_year_avg - prev_year_avg) / prev_year_avg if prev_year_avg > 0 else 0
            else:
                growth_rate = 0

            # Apply growth adjustment
            adjusted_avg = recent_avg * (1 + growth_rate)

            for h in range(1, horizon + 1):
                # Seasonal adjustment (simple month-over-month pattern)
                seasonal_factor = 1.0
                if len(group) >= 12:
                    month_idx = (group['year_month'].max().month + h - 1) % 12
                    seasonal_avg = group[group['year_month'].dt.month == month_idx + 1]['total_service_load'].mean()
                    overall_avg = group['total_service_load'].mean()
                    seasonal_factor = seasonal_avg / overall_avg if overall_avg > 0 else 1.0

                forecast_value = adjusted_avg * seasonal_factor
                forecasts.append({
                    'state': state,
                    'district': district,
                    'forecast_horizon': h,
                    'forecasted_load': max(0, forecast_value),
                    'method': 'moving_average_with_growth'
                })

        return pd.DataFrame(forecasts)
    
    def forecast_exponential_smoothing(self, df, horizon=None):
        """Exponential smoothing forecast"""
        if horizon is None:
            horizon = Config.FORECAST_HORIZON
        
        df = df.sort_values(['state', 'district', 'year_month'])
        forecasts = []
        
        for (state, district), group in df.groupby(['state', 'district']):
            if len(group) < 2:
                continue
            
            values = group['total_service_load'].values
            smoothed = values[0]
            
            for val in values[1:]:
                smoothed = self.alpha * val + (1 - self.alpha) * smoothed
            
            for h in range(1, horizon + 1):
                forecasts.append({
                    'state': state,
                    'district': district,
                    'forecast_horizon': h,
                    'forecasted_load': smoothed
                })
        
        return pd.DataFrame(forecasts)


# =============================================================================
# MODULE 5: PRIORITIZATION
# =============================================================================

class CampPrioritizer:
    """Ranks districts for camp placement"""
    
    def __init__(self):
        self.forecast_weight = Config.FORECAST_WEIGHT
        self.firs_weight = Config.FIRS_WEIGHT
        self.gap_weight = Config.GAP_WEIGHT
    
    def compute_priority_score(self, df, forecast_df):
        """Compute composite priority score"""
        latest_data = df.groupby(['state', 'district']).tail(1)
        forecast_3m = forecast_df[forecast_df['forecast_horizon'] == 3]
        
        priority_df = latest_data.merge(
            forecast_3m[['state', 'district', 'forecasted_load']], 
            on=['state', 'district'], 
            how='left'
        )
        
        # Normalize components
        priority_df['norm_forecast'] = self._normalize(priority_df['forecasted_load'])
        priority_df['norm_firs'] = 1 - priority_df['firs']
        priority_df['service_gap'] = priority_df['total_service_load'] / (priority_df['ma_6m'] + 1)
        priority_df['norm_gap'] = self._normalize(priority_df['service_gap'])
        
        # Composite priority score
        priority_df['priority_score'] = (
            priority_df['norm_forecast'] * self.forecast_weight +
            priority_df['norm_firs'] * self.firs_weight +
            priority_df['norm_gap'] * self.gap_weight
        )
        
        # Rank and categorize
        priority_df = priority_df.sort_values('priority_score', ascending=False)
        priority_df['rank'] = range(1, len(priority_df) + 1)
        priority_df['priority_tier'] = self._assign_tiers(priority_df['rank'], len(priority_df))
        
        return priority_df
    
    @staticmethod
    def _normalize(series):
        """Min-max normalization"""
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return series * 0
        return (series - min_val) / (max_val - min_val)
    
    @staticmethod
    def _assign_tiers(ranks, total):
        """Assign priority tiers based on rank"""
        return pd.cut(
            ranks,
            bins=[0, total*Config.CRITICAL_THRESHOLD, total*Config.HIGH_THRESHOLD, 
                  total*Config.MEDIUM_THRESHOLD, total],
            labels=['Critical', 'High', 'Medium', 'Low']
        )
    
    def generate_recommendations(self, priority_df, top_n=None):
        """Generate actionable camp placement recommendations"""
        if top_n is None:
            top_n = Config.TOP_N_RECOMMENDATIONS
        
        return priority_df.nsmallest(top_n, 'rank')[
            ['state', 'district', 'priority_score', 'priority_tier', 
             'forecasted_load', 'firs', 'total_service_load']
        ]


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

class FinancialInclusionScout:
    """Main framework orchestrator"""
    
    def __init__(self):
        self.cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.forecaster = ServiceLoadForecaster()
        self.prioritizer = CampPrioritizer()
    
    def run(self, enrolment_pattern=None, demographic_pattern=None, biometric_pattern=None):
        """Execute end-to-end framework"""

        print("\n" + "="*80)
        print("FINANCIAL INCLUSION SCOUT: AADHAAR SERVICE LOAD FORECASTING")
        print("="*80)

        # Step 1: Load data
        print("\n[1/6] Loading datasets...")
        enrolment = self._load_multiple_files(enrolment_pattern or 'api_data_aadhar_enrolment_*.csv')
        demographic = self._load_multiple_files(demographic_pattern or 'api_data_aadhar_demographic_*.csv')
        biometric = self._load_multiple_files(biometric_pattern or 'api_data_aadhar_biometric_*.csv')
        
        # Step 2: Clean data
        print("\n[2/6] Cleaning datasets...")
        enrolment_clean = self.cleaner.clean_dataset(enrolment, "Enrolment")
        demographic_clean = self.cleaner.clean_dataset(demographic, "Demographic")
        biometric_clean = self.cleaner.clean_dataset(biometric, "Biometric")
        
        # Step 3: Preprocess
        print("\n[3/6] Preprocessing and aggregation...")
        enrol_monthly = self.preprocessor.aggregate_to_monthly(enrolment_clean, 'registrar')
        demo_monthly = self.preprocessor.aggregate_to_monthly(demographic_clean, 'registrar')
        bio_monthly = self.preprocessor.aggregate_to_monthly(biometric_clean, 'registrar')
        
        # Step 4: Feature engineering
        print("\n[4/6] Engineering features...")
        service_load_df = self.engineer.compute_service_load(enrol_monthly, demo_monthly, bio_monthly)
        service_load_df = self.engineer.compute_inclusion_readiness(service_load_df)
        service_load_df = self.engineer.compute_trends(service_load_df)
        
        # Step 5: Forecasting
        print("\n[5/6] Forecasting service demand...")
        forecast_df = self.forecaster.forecast_moving_average(service_load_df)
        
        # Step 6: Prioritization
        print("\n[6/6] Generating priority rankings...")
        priority_df = self.prioritizer.compute_priority_score(service_load_df, forecast_df)
        recommendations = self.prioritizer.generate_recommendations(priority_df)
        
        # Save outputs
        self._save_outputs(service_load_df, forecast_df, priority_df, recommendations)
        
        # Display results
        self._display_results(recommendations)
        
        return service_load_df, forecast_df, priority_df, recommendations
    
    @staticmethod
    def _save_outputs(service_load_df, forecast_df, priority_df, recommendations):
        """Save all output files"""
        print("\n" + "="*80)
        print("SAVING OUTPUTS")
        print("="*80)
        
        service_load_df.to_csv('output_service_load.csv', index=False)
        forecast_df.to_csv('output_forecasts.csv', index=False)
        priority_df.to_csv('output_priority_rankings.csv', index=False)
        recommendations.to_csv('output_camp_recommendations.csv', index=False)
        
        print("\n✓ Service load data: output_service_load.csv")
        print("✓ Forecasts: output_forecasts.csv")
        print("✓ Priority rankings: output_priority_rankings.csv")
        print("✓ Camp recommendations: output_camp_recommendations.csv")
    
    @staticmethod
    def _display_results(recommendations):
        """Display top recommendations"""
        print("\n" + "="*80)
        print("TOP 10 PRIORITY DISTRICTS FOR CAMP PLACEMENT")
        print("="*80)
        print(recommendations.head(10).to_string(index=False))
        
        print("\n" + "="*80)
        print("FRAMEWORK EXECUTION COMPLETE")
        print("="*80)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    scout = FinancialInclusionScout()
    scout.run()


if __name__ == "__main__":
    main()
