#!/usr/bin/env python3
"""
Financial Inclusion Scout - UIDAI Data Hackathon
Clean, Concise, and Maintainable Implementation
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration"""
    
    VALID_STATES = {
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
        "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
        "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
        "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Puducherry", "Chandigarh",
        "Ladakh", "Jammu And Kashmir", "Andaman And Nicobar Islands",
        "Dadra And Nagar Haveli And Daman And Diu", "Lakshadweep"
    }
    
    STATE_MAPPING = {
        # City names misclassified as states
        "darbhanga": "Bihar", "puttenahalli": "Karnataka", "balanagar": "Telangana",
        "jaipur": "Rajasthan", "madanapalle": "Andhra Pradesh", "nagpur": "Maharashtra",
        "raja annamalai puram": "Tamil Nadu",
        # Legacy names
        "orissa": "Odisha", "uttaranchal": "Uttarakhand", "chattisgarh": "Chhattisgarh",
        "telengana": "Telangana", "pondicherry": "Puducherry",
        # Delhi variants
        "nct of delhi": "Delhi", "national capital territory of delhi": "Delhi",
        # J&K variants
        "jammu kashmir": "Jammu And Kashmir", "jammu and kashmir ut": "Jammu And Kashmir",
        # UT merger
        "dadra and nagar haveli": "Dadra And Nagar Haveli And Daman And Diu",
        "daman and diu": "Dadra And Nagar Haveli And Daman And Diu",
        # Others
        "andaman nicobar islands": "Andaman And Nicobar Islands"
    }
    
    # FIRS weights
    MOBILE_WEIGHT = 0.5
    DIGITAL_WEIGHT = 0.3
    COVERAGE_WEIGHT = 0.2


# =============================================================================
# DATA CLEANER
# =============================================================================

class DataCleaner:
    """Handles robust data cleaning"""
    
    @staticmethod
    def clean_text(text):
        """Standardize text fields"""
        if pd.isna(text):
            return text
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    
    @classmethod
    def clean_dataset(cls, df):
        """Complete data cleaning pipeline"""
        print(f"Initial records: {len(df):,}")
        
        # Standardize columns
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.rename(columns={"demo_age_17_": "demo_age_18_plus"})
        
        # Parse dates
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["date"] <= pd.Timestamp.now()]
        
        # Clean geographic fields
        df["state_clean"] = df["state"].apply(cls.clean_text)
        df["district_clean"] = df["district"].apply(cls.clean_text)
        
        # Standardize states
        df["state_clean"] = df["state_clean"].replace(Config.STATE_MAPPING)
        df = df[df["state_clean"].isin(Config.VALID_STATES)]
        
        # Clean numeric fields
        for col in ["pincode", "demo_age_5_17", "demo_age_18_plus"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Validate pincode
        df = df.dropna(subset=["state_clean", "district_clean", "pincode"])
        df = df[df["pincode"].astype(str).str.match(r"^\d{6}$")]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["date", "state_clean", "district_clean", "pincode"])
        
        # Create features
        df["total_enroll"] = df["demo_age_5_17"] + df["demo_age_18_plus"]
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["year_month"] = df["date"].dt.to_period('M')
        
        print(f"Cleaned records: {len(df):,} ({len(df)/len(df)*100:.1f}% retained)")
        print(f"Unique states: {df['state_clean'].nunique()}")
        
        return df


# =============================================================================
# FEATURE ENGINEER
# =============================================================================

class FeatureEngineer:
    """Computes FIRS and derived metrics"""
    
    @staticmethod
    def aggregate_signals(df):
        """Aggregate to district-year level"""
        return df.groupby(['state_clean', 'district_clean', 'year']).agg({
            'total_enroll': 'sum',
            'demo_age_5_17': 'sum',
            'demo_age_18_plus': 'sum',
            'pincode': 'nunique'
        }).reset_index()
    
    @staticmethod
    def compute_firs(df):
        """Calculate Financial Inclusion Readiness Score"""
        # Mobile readiness (adult enrollment rate)
        df['mobile_readiness'] = df['demo_age_18_plus'] / (df['total_enroll'] + 1)
        
        # Digital engagement (biometric proxy)
        df['biometric_updates'] = (df['total_enroll'] * 0.25).astype(int)
        df['digital_engagement'] = df['biometric_updates'] / (df['total_enroll'] + 1)
        
        # Coverage factor (pincode diversity)
        df['coverage_factor'] = np.log1p(df['pincode']) / 5
        
        # FIRS Score
        df['firs_score'] = (
            df['mobile_readiness'] * Config.MOBILE_WEIGHT +
            df['digital_engagement'] * Config.DIGITAL_WEIGHT +
            df['coverage_factor'] * Config.COVERAGE_WEIGHT
        ) * 100
        
        return df.round(2)
    
    @staticmethod
    def classify_readiness(df):
        """Classify regions by readiness"""
        high_thresh = df['firs_score'].quantile(0.75)
        low_thresh = df['firs_score'].quantile(0.25)
        
        df['readiness_zone'] = pd.cut(
            df['firs_score'],
            bins=[-np.inf, low_thresh, high_thresh, np.inf],
            labels=['Low Readiness', 'Medium Readiness', 'High Readiness']
        )
        
        return df
    
    @staticmethod
    def compute_trends(df):
        """Calculate year-over-year trends"""
        df = df.sort_values(['state_clean', 'district_clean', 'year'])
        df['firs_change'] = df.groupby(['state_clean', 'district_clean'])['firs_score'].pct_change()
        
        def classify_trend(change):
            if pd.isna(change):
                return 'New Region'
            elif change > 0.1:
                return 'Improving'
            elif change < -0.1:
                return 'Declining'
            else:
                return 'Stagnant'
        
        df['trend'] = df['firs_change'].apply(classify_trend)
        return df


# =============================================================================
# INSIGHT GENERATOR
# =============================================================================

class InsightGenerator:
    """Generates policy insights and recommendations"""
    
    @staticmethod
    def top_performers(df, n=10):
        """Identify top performing regions"""
        return df.nlargest(n, 'firs_score')[
            ['state_clean', 'district_clean', 'firs_score', 'total_enroll', 'readiness_zone']
        ]
    
    @staticmethod
    def priority_regions(df):
        """Identify high enrollment but low readiness regions"""
        return df[
            (df['total_enroll'] > df['total_enroll'].quantile(0.8)) &
            (df['readiness_zone'] == 'Low Readiness')
        ][['state_clean', 'district_clean', 'total_enroll', 'firs_score']]
    
    @staticmethod
    def state_summary(df):
        """State-wise aggregated summary"""
        return df.groupby('state_clean').agg({
            'firs_score': 'mean',
            'total_enroll': 'sum',
            'district_clean': 'nunique'
        }).round(2).sort_values('firs_score', ascending=False)
    
    @staticmethod
    def trend_summary(df):
        """Summarize trends"""
        return df['trend'].value_counts()


# =============================================================================
# MAIN FRAMEWORK
# =============================================================================

class FinancialInclusionScout:
    """Main orchestrator for Financial Inclusion Scout"""
    
    def __init__(self):
        self.df_clean = None
        self.firs_scores = None
        self.insights = {}
    
    def load_data(self, file_pattern="api_data_aadhar_demographic_*.csv"):
        """Load multiple CSV files"""
        import glob
        files = sorted(glob.glob(file_pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found matching: {file_pattern}")
        
        print(f"Loading {len(files)} files...")
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    
    def run(self):
        """Execute complete analysis pipeline"""
        print("\n" + "="*70)
        print("FINANCIAL INCLUSION SCOUT - COMPREHENSIVE ANALYSIS")
        print("="*70)
        
        # Step 1: Load and clean
        print("\n[1/5] Loading and cleaning data...")
        df_raw = self.load_data()
        self.df_clean = DataCleaner.clean_dataset(df_raw)
        
        # Step 2: Aggregate signals
        print("\n[2/5] Aggregating signals...")
        df_agg = FeatureEngineer.aggregate_signals(self.df_clean)
        print(f"Aggregated to {len(df_agg):,} district-year combinations")
        
        # Step 3: Compute FIRS
        print("\n[3/5] Computing FIRS scores...")
        df_firs = FeatureEngineer.compute_firs(df_agg)
        df_firs = FeatureEngineer.classify_readiness(df_firs)
        df_firs = FeatureEngineer.compute_trends(df_firs)
        self.firs_scores = df_firs
        print(f"Mean FIRS Score: {df_firs['firs_score'].mean():.2f}")
        
        # Step 4: Generate insights
        print("\n[4/5] Generating insights...")
        self.insights = {
            'top_performers': InsightGenerator.top_performers(df_firs),
            'priority_regions': InsightGenerator.priority_regions(df_firs),
            'state_summary': InsightGenerator.state_summary(df_firs),
            'trend_summary': InsightGenerator.trend_summary(df_firs)
        }
        
        # Step 5: Display results
        print("\n[5/5] Displaying results...")
        self._display_results()
        
        # Save outputs
        self._save_outputs()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        return self.firs_scores, self.insights
    
    def _display_results(self):
        """Display key results"""
        print("\n" + "-"*70)
        print("TOP 10 READY REGIONS")
        print("-"*70)
        print(self.insights['top_performers'].to_string(index=False))
        
        print("\n" + "-"*70)
        print("PRIORITY REGIONS (High Enrollment, Low Readiness)")
        print("-"*70)
        priority = self.insights['priority_regions']
        if len(priority) > 0:
            print(f"Found {len(priority)} priority regions")
            print(priority.head(10).to_string(index=False))
        else:
            print("No priority regions identified")
        
        print("\n" + "-"*70)
        print("TOP 10 STATES BY AVERAGE FIRS SCORE")
        print("-"*70)
        print(self.insights['state_summary'].head(10).to_string())
        
        print("\n" + "-"*70)
        print("TREND ANALYSIS")
        print("-"*70)
        print(self.insights['trend_summary'].to_string())
        
        print("\n" + "-"*70)
        print("READINESS ZONE DISTRIBUTION")
        print("-"*70)
        print(self.firs_scores['readiness_zone'].value_counts().to_string())
    
    def _save_outputs(self):
        """Save results to CSV"""
        self.firs_scores.to_csv('output_firs_scores.csv', index=False)
        self.insights['top_performers'].to_csv('output_top_performers.csv', index=False)
        self.insights['priority_regions'].to_csv('output_priority_regions.csv', index=False)
        self.insights['state_summary'].to_csv('output_state_summary.csv', index=False)
        
        print("\n✓ Saved: output_firs_scores.csv")
        print("✓ Saved: output_top_performers.csv")
        print("✓ Saved: output_priority_regions.csv")
        print("✓ Saved: output_state_summary.csv")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    scout = FinancialInclusionScout()
    firs_scores, insights = scout.run()
    return firs_scores, insights


if __name__ == "__main__":
    main()
