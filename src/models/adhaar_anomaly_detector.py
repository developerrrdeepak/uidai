
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class AadhaarAnomalyDetector:
    """
    Time-series anomaly detection system for Aadhaar enrolment and biometric update data.
    Detects abnormal changes at district level using Z-score method and Isolation Forest.
    """
    
    def __init__(self, rolling_window: int = 12, z_threshold: float = 2.0, 
                 consecutive_months: int = 2):
        """
        Initialize the anomaly detector.
        
        """
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold
        self.consecutive_months = consecutive_months
        self.signals = ['Enrolments', 'Updates', 'Child_0_5', 'Female']
        
    def prepare_data(self, enrolment_df: pd.DataFrame, update_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Data Preparation
        Convert dates to YYYY-MM format and aggregate by district and month.
        
        """
        # Process enrolment data
        enrolment_df = enrolment_df.copy()
        enrolment_df['Month'] = pd.to_datetime(enrolment_df['date']).dt.to_period('M').astype(str)
        
        # Aggregate enrolment data
        enrol_agg = enrolment_df.groupby(['district_clean', 'Month']).agg({
            'total_reg': 'sum',  # Total enrolments
            'demo_age_5_17': 'sum',  # Child enrolments (0-5)
            'demo_age_17_': 'sum'  # Adult enrolments
        }).reset_index()

        # Calculate female enrolments (assuming roughly equal distribution)
        enrol_agg['Female'] = (enrol_agg['total_reg'] * 0.52).astype(int)
        enrol_agg['Enrolments'] = enrol_agg['total_reg']
        enrol_agg['Child_0_5'] = enrol_agg['demo_age_5_17']

        enrol_agg = enrol_agg[['district_clean', 'Month', 'Enrolments', 'Female', 'Child_0_5']]
        enrol_agg.columns = ['District', 'Month', 'Enrolments', 'Female', 'Child_0_5']
        
        # Process update data
        update_df = update_df.copy()
        update_df['Month'] = pd.to_datetime(update_df['date']).dt.to_period('M').astype(str)

        # Aggregate update data
        update_agg = update_df.groupby(['district_clean', 'Month']).agg({
            'total_reg': 'sum'  # Total updates
        }).reset_index()

        update_agg.columns = ['District', 'Month', 'Updates']
        
        # Merge enrolment and update data
        merged_df = pd.merge(enrol_agg, update_agg, on=['District', 'Month'], how='outer')
        merged_df = merged_df.fillna(0)
        
        # Sort by district and month
        merged_df = merged_df.sort_values(['District', 'Month']).reset_index(drop=True)
        
        print(f"✓ Data prepared: {len(merged_df)} district-month records")
        print(f"  Districts: {merged_df['District'].nunique()}")
        print(f"  Time range: {merged_df['Month'].min()} to {merged_df['Month'].max()}")
        
        return merged_df
    
    def calculate_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Normal Behaviour Baseline
        Calculate rolling mean and std for each district using last 12 months.
        Each district is compared to itself over time (not to other districts).
      
        """
        result_df = df.copy()
        
        for signal in self.signals:
            # Calculate rolling statistics for each district
            result_df[f'{signal}_Mean'] = result_df.groupby('District')[signal].transform(
                lambda x: x.rolling(window=self.rolling_window, min_periods=1).mean().shift(1)
            )
            result_df[f'{signal}_Std'] = result_df.groupby('District')[signal].transform(
                lambda x: x.rolling(window=self.rolling_window, min_periods=1).std().shift(1)
            )
            
            # Replace NaN std with small value to avoid division by zero
            result_df[f'{signal}_Std'] = result_df[f'{signal}_Std'].fillna(1e-6)
            result_df[f'{signal}_Std'] = result_df[f'{signal}_Std'].replace(0, 1e-6)
        
        print(f"✓ Baseline calculated using {self.rolling_window}-month rolling window")
        
        return result_df
    
    def detect_anomalies_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Anomaly Detection using Z-Score Method
        Apply Z-score formula: Z = (Current_Value − Mean) / Std
        
        Decision Rules:
        - Z < −2: Significant drop (negative anomaly)
        - Z > +2: Unusual spike (positive anomaly)
        - −2 ≤ Z ≤ 2: Normal behaviour
        
        """
        result_df = df.copy()
        
        for signal in self.signals:
            # Calculate Z-score
            result_df[f'{signal}_ZScore'] = (
                (result_df[signal] - result_df[f'{signal}_Mean']) / result_df[f'{signal}_Std']
            )
            
            # Flag anomalies based on threshold
            result_df[f'{signal}_Anomaly'] = (
                (result_df[f'{signal}_ZScore'] < -self.z_threshold) | 
                (result_df[f'{signal}_ZScore'] > self.z_threshold)
            )
            
            # Classify anomaly type
            result_df[f'{signal}_AnomalyType'] = 'Normal'
            result_df.loc[result_df[f'{signal}_ZScore'] < -self.z_threshold, f'{signal}_AnomalyType'] = 'Drop'
            result_df.loc[result_df[f'{signal}_ZScore'] > self.z_threshold, f'{signal}_AnomalyType'] = 'Spike'
        
        total_anomalies = sum(result_df[f'{signal}_Anomaly'].sum() for signal in self.signals)
        print(f"✓ Z-score anomaly detection completed")
        print(f"  Total anomalies detected (before confirmation): {total_anomalies}")
        
        return result_df
    
    def confirm_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Anomaly Confirmation
        Confirm anomaly only if it persists for >= consecutive_months.
        This reduces false alarms and mimics real governance decision logic.
        
        """
        result_df = df.copy()
        
        for signal in self.signals:
            # Create confirmed anomaly column
            result_df[f'{signal}_ConfirmedAnomaly'] = False
            
            # For each district, check consecutive anomalies
            for district in result_df['District'].unique():
                district_mask = result_df['District'] == district
                district_data = result_df[district_mask].copy()
                
                # Calculate consecutive anomaly count
                district_data['consecutive'] = (
                    district_data[f'{signal}_Anomaly']
                    .groupby((district_data[f'{signal}_Anomaly'] != 
                             district_data[f'{signal}_Anomaly'].shift()).cumsum())
                    .transform('cumsum')
                )
                
                # Confirm if consecutive count >= threshold
                confirmed = (district_data[f'{signal}_Anomaly']) & (
                    district_data['consecutive'] >= self.consecutive_months
                )
                
                result_df.loc[district_mask, f'{signal}_ConfirmedAnomaly'] = confirmed.values
        
        total_confirmed = sum(result_df[f'{signal}_ConfirmedAnomaly'].sum() for signal in self.signals)
        print(f"✓ Anomaly confirmation completed")
        print(f"  Confirmed anomalies (>= {self.consecutive_months} consecutive months): {total_confirmed}")
        
        return result_df
    
    def calculate_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Severity Scoring
        Calculate severity = |Current − Expected| / Expected
        
        Classification:
        - < 20%: Low
        - 20–40%: Medium
        - > 40%: High
        
        """
        result_df = df.copy()
        
        for signal in self.signals:
            # Calculate severity percentage
            result_df[f'{signal}_Severity'] = (
                np.abs(result_df[signal] - result_df[f'{signal}_Mean']) / 
                (result_df[f'{signal}_Mean'] + 1e-6)  # Avoid division by zero
            ) * 100
            
            # Classify severity level
            result_df[f'{signal}_SeverityLevel'] = 'Normal'
            
            # Only assign severity levels to confirmed anomalies
            confirmed_mask = result_df[f'{signal}_ConfirmedAnomaly']
            
            result_df.loc[confirmed_mask & (result_df[f'{signal}_Severity'] < 20), 
                         f'{signal}_SeverityLevel'] = 'Low'
            result_df.loc[confirmed_mask & (result_df[f'{signal}_Severity'] >= 20) & 
                         (result_df[f'{signal}_Severity'] < 40), 
                         f'{signal}_SeverityLevel'] = 'Medium'
            result_df.loc[confirmed_mask & (result_df[f'{signal}_Severity'] >= 40), 
                         f'{signal}_SeverityLevel'] = 'High'
        
        print(f"✓ Severity scoring completed")
        
        return result_df
    
    def apply_isolation_forest(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """
        Step 6: Optional ML Validation using Isolation Forest
        Validate anomalies using multivariate approach.
        
        Features used:
        - Enrolment change %
        - Update change %
        - Child enrolment change %
        - Female enrolment change %
        
        """
        result_df = df.copy()
        
        # Prepare features: percentage changes for each signal
        for signal in self.signals:
            result_df[f'{signal}_Change%'] = (
                (result_df[signal] - result_df[f'{signal}_Mean']) / 
                (result_df[f'{signal}_Mean'] + 1e-6)
            ) * 100
        
        # Select features for Isolation Forest
        feature_cols = [f'{signal}_Change%' for signal in self.signals]
        
        # Remove rows with insufficient data
        valid_mask = result_df[feature_cols].notna().all(axis=1)
        valid_data = result_df[valid_mask].copy()
        
        if len(valid_data) > 0:
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Predict anomalies (-1 for anomaly, 1 for normal)
            predictions = iso_forest.fit_predict(valid_data[feature_cols])
            valid_data['IF_Anomaly'] = (predictions == -1)
            
            # Get anomaly scores
            valid_data['IF_Score'] = iso_forest.score_samples(valid_data[feature_cols])
            
            # Merge back to result_df
            result_df['IF_Anomaly'] = False
            result_df['IF_Score'] = 0.0
            result_df.loc[valid_mask, 'IF_Anomaly'] = valid_data['IF_Anomaly'].values
            result_df.loc[valid_mask, 'IF_Score'] = valid_data['IF_Score'].values
            
            print(f"✓ Isolation Forest validation completed")
            print(f"  ML-detected anomalies: {result_df['IF_Anomaly'].sum()}")
        else:
            result_df['IF_Anomaly'] = False
            result_df['IF_Score'] = 0.0
            print("⚠ Insufficient data for Isolation Forest")
        
        return result_df
    
    def generate_output_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 7: Generate Output Report
        Format: | District | Month | Signal | Anomaly | Severity |
        """
        report_records = []
        
        for _, row in df.iterrows():
            for signal in self.signals:
                if row[f'{signal}_ConfirmedAnomaly']:
                    report_records.append({
                        'District': row['District'],
                        'Month': row['Month'],
                        'Signal': signal,
                        'Anomaly': 'Yes',
                        'AnomalyType': row[f'{signal}_AnomalyType'],
                        'Severity': row[f'{signal}_SeverityLevel'],
                        'SeverityScore': round(row[f'{signal}_Severity'], 2),
                        'ZScore': round(row[f'{signal}_ZScore'], 2),
                        'CurrentValue': int(row[signal]),
                        'ExpectedValue': round(row[f'{signal}_Mean'], 2),
                        'IF_Validated': row['IF_Anomaly'] if 'IF_Anomaly' in row else False
                    })
        
        report_df = pd.DataFrame(report_records)
        
        if len(report_df) > 0:
            print(f"\n✓ Output report generated")
            print(f"  Total confirmed anomaly records: {len(report_df)}")
            print(f"  Affected districts: {report_df['District'].nunique()}")
            print(f"\nSeverity breakdown:")
            print(report_df['Severity'].value_counts())
        else:
            print("\n✓ No confirmed anomalies detected")
        
        return report_df
    
    def run_full_analysis(self, enrolment_df: pd.DataFrame, 
                         update_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete anomaly detection pipeline.
        
        """
        print("=" * 70)
        print("AADHAAR TIME-SERIES ANOMALY DETECTION SYSTEM")
        print("=" * 70)
        
        # Step 1: Data Preparation
        print("\n[STEP 1] Data Preparation")
        prepared_df = self.prepare_data(enrolment_df, update_df)
        
        # Step 2: Calculate Baseline
        print("\n[STEP 2] Normal Behaviour Baseline")
        baseline_df = self.calculate_baseline(prepared_df)
        
        # Step 3: Detect Anomalies (Z-Score)
        print("\n[STEP 3] Anomaly Detection (Z-Score Method)")
        anomaly_df = self.detect_anomalies_zscore(baseline_df)
        
        # Step 4: Confirm Anomalies
        print("\n[STEP 4] Anomaly Confirmation")
        confirmed_df = self.confirm_anomalies(anomaly_df)
        
        # Step 5: Calculate Severity
        print("\n[STEP 5] Severity Scoring")
        severity_df = self.calculate_severity(confirmed_df)
        
        # Step 6: Isolation Forest Validation
        print("\n[STEP 6] ML Validation (Isolation Forest)")
        final_df = self.apply_isolation_forest(severity_df)
        
        # Step 7: Generate Report
        print("\n[STEP 7] Output Report Generation")
        report_df = self.generate_output_report(final_df)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        
        return final_df, report_df
    
    def get_summary_statistics(self, report_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics from the anomaly report.
        
        """
        if len(report_df) == 0:
            return {
                'total_anomalies': 0,
                'affected_districts': 0,
                'severity_breakdown': {},
                'signal_breakdown': {},
                'anomaly_type_breakdown': {}
            }
        
        summary = {
            'total_anomalies': len(report_df),
            'affected_districts': report_df['District'].nunique(),
            'severity_breakdown': report_df['Severity'].value_counts().to_dict(),
            'signal_breakdown': report_df['Signal'].value_counts().to_dict(),
            'anomaly_type_breakdown': report_df['AnomalyType'].value_counts().to_dict(),
            'ml_validated_count': report_df['IF_Validated'].sum()
        }
        
        return summary