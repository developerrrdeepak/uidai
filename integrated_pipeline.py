"""
🏆 UIDAI INTEGRATED PIPELINE - ALL 3 MODELS COMBINED
=================================================

Flow: UIDAI Data → Model 1 (Anomaly) → Model 2 (Risk Score) → Model 3 (Classification) → Final Alert
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Model1AnomalyDetector:
    """Model 1: Anomaly Detection - kab problem aayi"""
    
    def detect_anomalies(self, df):
        """Detect anomalies in UIDAI data"""
        results = df.copy()
        
        # Enrolment anomalies
        results['enrolment_anomaly'] = (
            (results['enrolment_pct_change'] > 30) | 
            (results['enrolment_pct_change'] < -30)
        )
        
        # Update anomalies  
        results['update_anomaly'] = (
            (results['update_pct_change'] < -25)
        )
        
        # Child anomalies
        results['child_anomaly'] = (
            (results['child_pct_change'] < -20)
        )
        
        # Female anomalies
        results['female_anomaly'] = (
            (results['female_pct_change'] < -15)
        )
        
        # Sudden spikes/drops
        results['sudden_spike'] = results['enrolment_pct_change'] > 40
        results['sudden_large_drop'] = results['enrolment_pct_change'] < -40
        
        return results


class Model2RiskScoring:
    """Model 2: Risk Scoring - kaun sa district weak hai"""
    
    def calculate_risk_score(self, df):
        """Calculate risk scores for districts"""
        results = df.copy()
        
        # Anomaly severity (0-1)
        anomaly_count = (
            df['enrolment_anomaly'].astype(int) +
            df['update_anomaly'].astype(int) +
            df['child_anomaly'].astype(int) +
            df['female_anomaly'].astype(int)
        )
        results['anomaly_severity'] = np.clip(anomaly_count / 4.0, 0, 1)
        
        # Structural risk (based on percentage changes)
        enrol_risk = np.clip(abs(df['enrolment_pct_change']) / 50.0, 0, 1)
        update_risk = np.clip(abs(df['update_pct_change']) / 40.0, 0, 1)
        child_risk = np.clip(abs(df['child_pct_change']) / 30.0, 0, 1)
        female_risk = np.clip(abs(df['female_pct_change']) / 25.0, 0, 1)
        
        results['structural_risk_score'] = (
            enrol_risk * 0.4 + update_risk * 0.3 + 
            child_risk * 0.2 + female_risk * 0.1
        )
        
        # Final risk score
        results['risk_score'] = np.clip(
            results['anomaly_severity'] * 0.6 + 
            results['structural_risk_score'] * 0.4, 
            0, 1
        )
        
        return results


class Model3RiskClassifier:
    """Model 3: Risk Classification - kyon problem aayi"""
    
    def __init__(self):
        self.risk_types = {
            'Child Exclusion Risk': 'Anganwadi-based enrolment drives',
            'Gender Gap Risk': 'Women-only enrolment camps',
            'Update Failure Risk': 'Mobile update camps',
            'Administrative Disruption Risk': 'Staff reallocation',
            'Migration / Crisis Shock Risk': 'Emergency portable services'
        }
    
    def classify_risk(self, df):
        """Classify risk type and provide recommendations"""
        results = df.copy()
        
        # Initialize columns
        results['risk_type'] = 'Low Risk'
        results['reason'] = 'Normal operations'
        results['action'] = 'Continue monitoring'
        results['alert'] = 'No'
        results['risk_level'] = 'Low'
        
        for idx, row in results.iterrows():
            risk_type, reason, action = self._determine_risk_type(row)
            
            results.loc[idx, 'risk_type'] = risk_type
            results.loc[idx, 'reason'] = reason
            results.loc[idx, 'action'] = action
            
            # Set risk level and alert
            if row['risk_score'] >= 0.7:
                results.loc[idx, 'risk_level'] = 'High'
                results.loc[idx, 'alert'] = 'Yes'
            elif row['risk_score'] >= 0.4:
                results.loc[idx, 'risk_level'] = 'Medium'
                results.loc[idx, 'alert'] = 'Monitor'
            else:
                results.loc[idx, 'risk_level'] = 'Low'
                results.loc[idx, 'alert'] = 'No'
        
        return results
    
    def _determine_risk_type(self, row):
        """Determine specific risk type based on patterns"""
        
        # Migration/Crisis (highest priority)
        if row['sudden_spike'] or row['sudden_large_drop']:
            if row['sudden_spike']:
                return ('Migration / Crisis Shock Risk', 
                       f"Sudden spike: {row['enrolment_pct_change']:.0f}% increase", 
                       'Emergency portable services')
            else:
                return ('Migration / Crisis Shock Risk', 
                       f"Mass exodus: {row['enrolment_pct_change']:.0f}% drop", 
                       'Emergency portable services')
        
        # Child Exclusion
        if row['child_anomaly']:
            return ('Child Exclusion Risk', 
                   f"Child enrolments down {abs(row['child_pct_change']):.0f}%", 
                   'Anganwadi-based enrolment drives')
        
        # Gender Gap
        if row['female_anomaly']:
            return ('Gender Gap Risk', 
                   f"Female enrolments down {abs(row['female_pct_change']):.0f}%", 
                   'Women-only enrolment camps')
        
        # Update Failure
        if row['update_anomaly']:
            return ('Update Failure Risk', 
                   f"Updates down {abs(row['update_pct_change']):.0f}%", 
                   'Mobile update camps')
        
        # Administrative Issues
        if row['enrolment_anomaly']:
            return ('Administrative Disruption Risk', 
                   f"Enrolments down {abs(row['enrolment_pct_change']):.0f}%", 
                   'Staff reallocation')
        
        return ('Low Risk', 'Normal operations', 'Continue monitoring')


class UIDaiIntegratedPipeline:
    """🏆 MAIN PIPELINE - Integrates all 3 models"""
    
    def __init__(self):
        self.model1 = Model1AnomalyDetector()
        self.model2 = Model2RiskScoring()
        self.model3 = Model3RiskClassifier()
    
    def process_pipeline(self, raw_data):
        """
        Complete pipeline: Raw Data → Final Alert Table
        """
        print("Starting UIDAI Integrated Pipeline...")
        
        # Step 1: Anomaly Detection
        print("Step 1: Anomaly Detection...")
        anomaly_data = self.model1.detect_anomalies(raw_data)
        
        # Step 2: Risk Scoring
        print("Step 2: Risk Scoring...")
        risk_data = self.model2.calculate_risk_score(anomaly_data)
        
        # Step 3: Risk Classification
        print("Step 3: Risk Classification...")
        final_results = self.model3.classify_risk(risk_data)
        
        # Step 4: Generate Final Alert Table
        print("Step 4: Generating Final Alert Table...")
        alert_table = self._create_alert_table(final_results)
        
        print("Pipeline Complete!")
        return alert_table
    
    def _create_alert_table(self, results):
        """Create the hero output table"""
        alert_table = pd.DataFrame({
            'District': results['district'],
            'Risk Score': results['risk_score'].round(2),
            'Risk Level': results['risk_level'],
            'Alert': results['alert'],
            'Risk Type': results['risk_type'],
            'Reason': results['reason'],
            'Action': results['action']
        })
        
        # Sort by risk score (highest first)
        alert_table = alert_table.sort_values('Risk Score', ascending=False)
        alert_table = alert_table.reset_index(drop=True)
        
        return alert_table


def create_sample_uidai_data(n_districts=10):
    """Create sample UIDAI data for testing"""
    
    districts = [
        'Kalahandi', 'Koraput', 'Rayagada', 'Gajapati', 'Kandhamal',
        'Bolangir', 'Nuapada', 'Bargarh', 'Jharsuguda', 'Sambalpur',
        'Dhenkanal', 'Angul', 'Deogarh', 'Sundargarh', 'Keonjhar'
    ][:n_districts]
    
    np.random.seed(42)
    
    data = {
        'district': districts,
        'enrolment_pct_change': np.random.normal(0, 20, n_districts),
        'update_pct_change': np.random.normal(-5, 15, n_districts),
        'child_pct_change': np.random.normal(2, 18, n_districts),
        'female_pct_change': np.random.normal(1, 12, n_districts)
    }
    
    # Add some crisis scenarios
    if n_districts >= 3:
        data['enrolment_pct_change'][0] = -38  # Kalahandi crisis
        data['update_pct_change'][0] = -35
        data['child_pct_change'][1] = -25      # Koraput child exclusion
        data['female_pct_change'][2] = -20     # Rayagada gender gap
    
    return pd.DataFrame(data)


def main():
    """DEMO: Complete Pipeline in Action"""
    
    print("\n" + "=" * 80)
    print("UIDAI INTEGRATED PIPELINE - COMPLETE DEMO")
    print("=" * 80 + "\n")
    
    # Create sample data
    print("Creating sample UIDAI data...")
    raw_data = create_sample_uidai_data(n_districts=8)
    print(f"Generated data for {len(raw_data)} districts\n")
    
    # Run pipeline
    pipeline = UIDaiIntegratedPipeline()
    final_table = pipeline.process_pipeline(raw_data)
    
    # Display results
    print("\n" + "FINAL ALERT TABLE (HERO OUTPUT)")
    print("=" * 80)
    print(final_table.to_string(index=False))
    
    # High priority alerts
    high_alerts = final_table[final_table['Alert'] == 'Yes']
    if len(high_alerts) > 0:
        print(f"\nHIGH PRIORITY ALERTS: {len(high_alerts)} districts need immediate action!")
        for _, row in high_alerts.iterrows():
            print(f"   • {row['District']}: {row['Risk Type']} - {row['Action']}")
    
    print(f"\nSUMMARY:")
    print(f"   Total Districts: {len(final_table)}")
    print(f"   High Risk: {len(final_table[final_table['Risk Level'] == 'High'])}")
    print(f"   Medium Risk: {len(final_table[final_table['Risk Level'] == 'Medium'])}")
    print(f"   Low Risk: {len(final_table[final_table['Risk Level'] == 'Low'])}")
    
    print("\nPipeline ready for production!")
    return final_table


if __name__ == "__main__":
    main()