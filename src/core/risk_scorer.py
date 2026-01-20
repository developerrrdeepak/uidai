"""Risk Scoring Module"""
import numpy as np

class RiskScorer:
    """Calculates risk scores for districts"""
    
    def calculate(self, df):
        """Calculate composite risk scores"""
        result = df.copy()
        
        # Anomaly severity
        anomaly_count = (
            df['enrolment_anomaly'].astype(int) +
            df['update_anomaly'].astype(int) +
            df['child_anomaly'].astype(int) +
            df['female_anomaly'].astype(int)
        )
        result['anomaly_severity'] = np.clip(anomaly_count / 4.0, 0, 1)
        
        # Structural risk
        enrol_risk = np.clip(abs(df['enrolment_pct_change']) / 50.0, 0, 1)
        update_risk = np.clip(abs(df['update_pct_change']) / 40.0, 0, 1)
        child_risk = np.clip(abs(df['child_pct_change']) / 30.0, 0, 1)
        female_risk = np.clip(abs(df['female_pct_change']) / 25.0, 0, 1)
        
        result['structural_risk_score'] = (
            enrol_risk * 0.4 + update_risk * 0.3 + 
            child_risk * 0.2 + female_risk * 0.1
        )
        
        # Final risk score
        result['risk_score'] = np.clip(
            result['anomaly_severity'] * 0.6 + 
            result['structural_risk_score'] * 0.4, 
            0, 1
        )
        
        return result
