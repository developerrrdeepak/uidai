"""Risk Classification Module"""
import pandas as pd

class RiskClassifier:
    """Classifies risk types and generates recommendations"""
    
    def classify(self, df):
        """Classify risk types for all districts"""
        results = []
        for _, row in df.iterrows():
            results.append(self._classify_single(row))
        return pd.DataFrame(results)
    
    def _classify_single(self, row):
        """Classify single district"""
        risk_type, reason, action = self._determine_risk_type(row)
        severity = self._calculate_severity(row)
        
        return {
            'District': row['district'],
            'Risk Score': round(row['risk_score'], 2),
            'Risk Level': severity,
            'Alert': 'Yes' if severity == 'High' else ('Monitor' if severity == 'Medium' else 'No'),
            'Risk Type': risk_type,
            'Reason': reason,
            'Action': action
        }
    
    def _determine_risk_type(self, row):
        """Determine risk type based on anomalies"""
        if row.get('sudden_spike', False) or row.get('sudden_large_drop', False):
            change = row['enrolment_pct_change']
            return ('Migration / Crisis Shock Risk', 
                   f"Sudden {'spike' if change > 0 else 'drop'}: {abs(change):.0f}%",
                   'Emergency portable services')
        
        if row.get('child_anomaly', False):
            return ('Child Exclusion Risk',
                   f"Child enrolments down {abs(row['child_pct_change']):.0f}%",
                   'Anganwadi-based enrolment drives')
        
        if row.get('female_anomaly', False):
            return ('Gender Gap Risk',
                   f"Female enrolments down {abs(row['female_pct_change']):.0f}%",
                   'Women-only enrolment camps')
        
        if row.get('update_anomaly', False):
            return ('Update Failure Risk',
                   f"Updates down {abs(row['update_pct_change']):.0f}%",
                   'Mobile update camps')
        
        if row.get('enrolment_anomaly', False):
            return ('Administrative Disruption Risk',
                   f"Enrolments down {abs(row['enrolment_pct_change']):.0f}%",
                   'Staff reallocation')
        
        return ('Low Risk', 'Normal operations', 'Continue monitoring')
    
    def _calculate_severity(self, row):
        """Calculate severity level"""
        score = row.get('risk_score', 0)
        if score >= 0.5:
            return 'High'
        elif score >= 0.35:
            return 'Medium'
        return 'Low'
