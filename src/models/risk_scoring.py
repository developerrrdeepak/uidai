"""
Risk Scoring Engine for Financial Inclusion Risk Scoring Model
Implements weighted composite risk index with policy-driven weights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json


class RiskScoringEngine:
    """Calculate district-level financial inclusion risk scores"""
    
    # Policy-driven weights (must sum to 1.0)
    WEIGHTS = {
        'coverage': 0.30,   # Future inclusion foundation
        'child': 0.25,      # Long-term welfare access
        'gender': 0.20,     # Equity and social barriers
        'update': 0.15,     # DBT continuity
        'momentum': 0.10    # Trend signal
    }
    
    # Risk categorization thresholds
    RISK_THRESHOLDS = {
        'low': (0.00, 0.33),
        'medium': (0.34, 0.66),
        'high': (0.67, 1.00)
    }
    
    def __init__(self):
        """Initialize the risk scoring engine"""
        self.risk_scores = None
        
    def calculate_composite_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite risk score
        
        Risk_Score = 0.30 × Coverage_Risk
                   + 0.25 × Child_Risk
                   + 0.20 × Gender_Risk
                   + 0.15 × Update_Risk
                   + 0.10 × Momentum_Risk
        
        Args:
            df: DataFrame with risk-aligned indicators
            
        Returns:
            Series with risk scores [0, 1]
        """
        risk_score = (
            self.WEIGHTS['coverage'] * df['coverage_risk'] +
            self.WEIGHTS['child'] * df['child_risk'] +
            self.WEIGHTS['gender'] * df['gender_risk'] +
            self.WEIGHTS['update'] * df['update_risk'] +
            self.WEIGHTS['momentum'] * df['momentum_risk']
        )
        
        # Ensure scores are in [0, 1] range
        risk_score = risk_score.clip(0, 1)
        
        return risk_score
    
    def categorize_risk(self, risk_score: float) -> str:
        """
        Categorize risk score into Low/Medium/High
        
        Risk Score Range    Category
        0.00 – 0.33        Low Risk
        0.34 – 0.66        Medium Risk
        0.67 – 1.00        High Risk
        
        Args:
            risk_score: Risk score value [0, 1]
            
        Returns:
            Risk category string
        """
        if risk_score <= self.RISK_THRESHOLDS['low'][1]:
            return 'Low'
        elif risk_score <= self.RISK_THRESHOLDS['medium'][1]:
            return 'Medium'
        else:
            return 'High'
    
    def identify_top_drivers(self, row: pd.Series, top_n: int = 2) -> List[Tuple[str, float]]:
        """
        Identify top risk drivers for a district
        
        Args:
            row: DataFrame row with risk indicators
            top_n: Number of top drivers to return
            
        Returns:
            List of (driver_name, risk_value) tuples
        """
        drivers = {
            'Low Coverage': row['coverage_risk'],
            'Low Child Ratio': row['child_risk'],
            'Gender Gap': row['gender_risk'],
            'Update Delay': row['update_risk'],
            'Negative Momentum': row['momentum_risk']
        }
        
        # Sort by risk value (descending)
        sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_drivers[:top_n]
    
    def generate_policy_recommendation(self, risk_category: str, top_drivers: List[Tuple[str, float]]) -> str:
        """
        Generate policy recommendation based on risk profile
        
        Args:
            risk_category: Risk category (Low/Medium/High)
            top_drivers: List of top risk drivers
            
        Returns:
            Policy recommendation string
        """
        recommendations = {
            'Low Coverage': 'Expand Aadhaar enrollment centers and mobile units',
            'Low Child Ratio': 'Prioritize child enrollment drives in schools and anganwadis',
            'Gender Gap': 'Address social barriers through women-focused outreach programs',
            'Update Delay': 'Establish biometric update camps and simplify update process',
            'Negative Momentum': 'Investigate causes of enrollment decline and implement corrective measures'
        }
        
        if risk_category == 'High':
            priority = 'URGENT INTERVENTION REQUIRED'
        elif risk_category == 'Medium':
            priority = 'MODERATE INTERVENTION NEEDED'
        else:
            priority = 'MONITORING RECOMMENDED'
        
        driver_recommendations = [recommendations.get(driver[0], 'General monitoring') 
                                 for driver in top_drivers]
        
        return f"{priority}: {'; '.join(driver_recommendations)}"
    
    def compute_district_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute complete risk assessment for all districts
        
        Args:
            df: DataFrame with processed risk indicators
            
        Returns:
            DataFrame with risk scores, categories, drivers, and recommendations
        """
        result_df = df.copy()
        
        # Calculate risk scores
        result_df['risk_score'] = self.calculate_composite_risk_score(df)
        
        # Categorize risk levels
        result_df['risk_category'] = result_df['risk_score'].apply(self.categorize_risk)
        
        # Identify top drivers
        result_df['top_drivers'] = result_df.apply(
            lambda row: self.identify_top_drivers(row), axis=1
        )
        
        # Format top drivers as string
        result_df['top_drivers_text'] = result_df['top_drivers'].apply(
            lambda drivers: ', '.join([f"{d[0]} ({d[1]:.2f})" for d in drivers])
        )
        
        # Generate policy recommendations
        result_df['policy_recommendation'] = result_df.apply(
            lambda row: self.generate_policy_recommendation(
                row['risk_category'], row['top_drivers']
            ), axis=1
        )
        
        self.risk_scores = result_df
        
        return result_df
    
    def get_risk_summary(self) -> Dict:
        """
        Generate summary statistics of risk assessment
        
        Returns:
            Dictionary with summary statistics
        """
        if self.risk_scores is None:
            raise ValueError("No risk scores computed. Run compute_district_risk_scores first.")
        
        summary = {
            'total_districts': len(self.risk_scores),
            'high_risk_count': len(self.risk_scores[self.risk_scores['risk_category'] == 'High']),
            'medium_risk_count': len(self.risk_scores[self.risk_scores['risk_category'] == 'Medium']),
            'low_risk_count': len(self.risk_scores[self.risk_scores['risk_category'] == 'Low']),
            'average_risk_score': float(self.risk_scores['risk_score'].mean()),
            'highest_risk_district': self.risk_scores.loc[self.risk_scores['risk_score'].idxmax(), 'district'],
            'highest_risk_score': float(self.risk_scores['risk_score'].max()),
            'lowest_risk_district': self.risk_scores.loc[self.risk_scores['risk_score'].idxmin(), 'district'],
            'lowest_risk_score': float(self.risk_scores['risk_score'].min())
        }
        
        return summary
    
    def get_top_risk_districts(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N highest risk districts
        
        Args:
            n: Number of districts to return
            
        Returns:
            DataFrame with top risk districts
        """
        if self.risk_scores is None:
            raise ValueError("No risk scores computed. Run compute_district_risk_scores first.")
        
        top_districts = self.risk_scores.nlargest(n, 'risk_score')[
            ['district', 'state', 'risk_score', 'risk_category', 'top_drivers_text', 'policy_recommendation']
        ]
        
        return top_districts
    
    def export_results(self, output_path: str, format: str = 'json'):
        """
        Export risk assessment results
        
        Args:
            output_path: Path to save the file
            format: 'json' or 'csv'
        """
        if self.risk_scores is None:
            raise ValueError("No risk scores computed. Run compute_district_risk_scores first.")
        
        # Select relevant columns for export
        export_columns = ['district', 'state', 'risk_score', 'risk_category', 
                         'coverage_risk', 'child_risk', 'gender_risk', 
                         'update_risk', 'momentum_risk', 'top_drivers_text', 
                         'policy_recommendation']
        
        export_df = self.risk_scores[export_columns]
        
        if format == 'json':
            export_df.to_json(output_path, orient='records', indent=2)
        elif format == 'csv':
            export_df.to_csv(output_path, index=False)
        else:
            raise ValueError("Format must be 'json' or 'csv'")
    
    def explain_methodology(self) -> str:
        """
        Generate explanation of the risk scoring methodology
        
        Returns:
            Detailed methodology explanation
        """
        explanation = f"""
FINANCIAL INCLUSION RISK SCORING METHODOLOGY

1. WEIGHTED COMPOSITE RISK INDEX

The risk score is calculated using a policy-driven weighted composite index:

Risk_Score = {self.WEIGHTS['coverage']:.2f} × Coverage_Risk
           + {self.WEIGHTS['child']:.2f} × Child_Risk
           + {self.WEIGHTS['gender']:.2f} × Gender_Risk
           + {self.WEIGHTS['update']:.2f} × Update_Risk
           + {self.WEIGHTS['momentum']:.2f} × Momentum_Risk

2. WEIGHT JUSTIFICATION

- Coverage (30%): Foundation for future inclusion - measures basic access
- Child Ratio (25%): Long-term welfare indicator - predicts future exclusion
- Gender Gap (20%): Equity measure - identifies social barriers
- Update Ratio (15%): DBT continuity - ensures benefit delivery
- Momentum (10%): Trend signal - detects emerging problems

3. RISK CATEGORIZATION

- Low Risk:    0.00 - 0.33 (Monitoring recommended)
- Medium Risk: 0.34 - 0.66 (Moderate intervention needed)
- High Risk:   0.67 - 1.00 (Urgent intervention required)

4. INDICATOR CALCULATIONS

a) Enrolment Coverage = district_enrolments / state_average
b) Child Inclusion = child_0_5_enrolments / total_enrolments
c) Gender Gap = |male - female| / total_enrolments
d) Update Readiness = biometric_updates / total_enrolments
e) Enrolment Momentum = (current - previous) / previous

All indicators are normalized to [0,1] and aligned to risk direction.

5. MODEL TRANSPARENCY

This is an explainable, policy-grade model designed for:
- Clear interpretation by administrators
- Actionable insights for intervention
- Transparent decision-making process
- Accountability in resource allocation
"""
        return explanation


if __name__ == "__main__":
    # Load and process cleaned CSV data
    import pandas as pd
    import os

    print("Loading cleaned CSV data...")

    # Load demographic data
    demo_file = "data/cleaned/aadhaar_demographic_cleaned.csv"
    if os.path.exists(demo_file):
        demo_df = pd.read_csv(demo_file)
        print(f"✓ Loaded {len(demo_df)} demographic records")
    else:
        print(f"❌ File not found: {demo_file}")
        exit(1)

    # Load biometric data
    bio_file = "data/cleaned/aadhaar_biometric_cleaned.csv"
    if os.path.exists(bio_file):
        bio_df = pd.read_csv(bio_file)
        print(f"Loaded {len(bio_df)} biometric records")
    else:
        print(f"❌ File not found: {bio_file}")
        exit(1)

    # Aggregate data by district for risk scoring
    print("\nAggregating data by district...")

    # Demographic aggregation
    demo_agg = demo_df.groupby('district_clean').agg({
        'total_reg': 'sum',
        'demo_age_5_17': 'sum',
        'demo_age_17_': 'sum',
        'state_clean': 'first'
    }).reset_index()

    # Biometric aggregation
    bio_agg = bio_df.groupby('district_clean').agg({
        'bio_age_5_17': 'sum',
        'bio_age_17_': 'sum'
    }).reset_index()
    bio_agg['total_reg'] = bio_agg['bio_age_5_17'] + bio_agg['bio_age_17_']

    # Merge demographic and biometric data
    district_data = pd.merge(
        demo_agg,
        bio_agg,
        on='district_clean',
        how='outer',
        suffixes=('_demo', '_bio')
    ).fillna(0)

    # Transform to expected format for risk scoring
    district_data = district_data.rename(columns={
        'district_clean': 'district',
        'state_clean': 'state',
        'total_reg_demo': 'total_enrolments',
        'demo_age_5_17': 'child_0_5_enrolments',
        'total_reg_bio': 'biometric_updates'
    })

    # Estimate gender distribution (assuming roughly equal for demo purposes)
    district_data['male_enrolments'] = (district_data['total_enrolments'] * 0.52).astype(int)
    district_data['female_enrolments'] = district_data['total_enrolments'] - district_data['male_enrolments']

    # Add momentum data (simplified - using current vs previous period)
    district_data['current_month_enrolments'] = district_data['total_enrolments'] * 0.1  # Assume 10% recent
    district_data['previous_month_enrolments'] = district_data['total_enrolments'] * 0.09  # Assume 9% previous

    print(f"✓ Processed {len(district_data)} districts")

    # Process indicators
    from src.core.data_processor import DataProcessor
    processor = DataProcessor()
    processed_data = processor.process_all_indicators(district_data)

    # Calculate risk scores
    print("\nCalculating risk scores...")
    engine = RiskScoringEngine()
    risk_results = engine.compute_district_risk_scores(processed_data)

    # Display summary
    print("\n" + "="*80)
    print("RISK ASSESSMENT SUMMARY")
    print("="*80)
    summary = engine.get_risk_summary()
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Display top 10 high-risk districts
    print("\n" + "="*80)
    print("TOP 10 HIGH-RISK DISTRICTS")
    print("="*80)
    top_districts = engine.get_top_risk_districts(n=10)
    print(top_districts.to_string(index=False))

    # Export results
    print("\nExporting results...")
    engine.export_results('risk_assessment_results.json', format='json')
    print("Results exported successfully!")

    # Print methodology
    print("\n" + engine.explain_methodology())

    # Display EDA charts
    print("\nDisplaying EDA Charts...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        # Display demographic EDA chart
        if os.path.exists('outputs/images/eda_charts.png'):
            img = mpimg.imread('outputs/images/eda_charts.png')
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Demographic EDA Charts')
            plt.show()

        # Display biometric EDA chart
        if os.path.exists('outputs/images/eda_charts_biometric.png'):
            img = mpimg.imread('outputs/images/eda_charts_biometric.png')
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Biometric EDA Charts')
            plt.show()

        print("EDA charts displayed successfully!")
    except ImportError:
        print("Matplotlib not available for displaying charts. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error displaying charts: {e}")
