"""Anomaly Detection Module"""

class AnomalyDetector:
    """Detects anomalies in UIDAI data"""
    
    def detect(self, df):
        """Detect anomalies across all metrics"""
        result = df.copy()
        
        result['enrolment_anomaly'] = (
            (result['enrolment_pct_change'] > 30) | 
            (result['enrolment_pct_change'] < -30)
        )
        result['update_anomaly'] = result['update_pct_change'] < -25
        result['child_anomaly'] = result['child_pct_change'] < -20
        result['female_anomaly'] = result['female_pct_change'] < -15
        result['sudden_spike'] = result['enrolment_pct_change'] > 40
        result['sudden_large_drop'] = result['enrolment_pct_change'] < -40
        
        return result
