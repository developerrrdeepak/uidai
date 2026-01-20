"""Policy Recommendation Engine"""
from datetime import datetime

class PolicyEngine:
    """Generate policy recommendations for risk mitigation"""
    
    POLICY_MATRIX = {
        'Child Exclusion Risk': {
            'action': 'Anganwadi-based enrolment drives',
            'timeline': '2-4 weeks',
            'budget': '₹2-5 lakhs',
            'dept': 'Women & Child Development + UIDAI'
        },
        'Gender Gap Risk': {
            'action': 'Women-only enrolment camps',
            'timeline': '3-6 weeks',
            'budget': '₹3-7 lakhs',
            'dept': 'Women & Child Development + UIDAI'
        },
        'Update Failure Risk': {
            'action': 'Mobile update camps',
            'timeline': '1-3 weeks',
            'budget': '₹1-3 lakhs',
            'dept': 'UIDAI + District Administration'
        },
        'Administrative Disruption Risk': {
            'action': 'Staff reallocation',
            'timeline': '1-2 weeks',
            'budget': '₹5-10 lakhs',
            'dept': 'UIDAI + State Government'
        },
        'Migration / Crisis Shock Risk': {
            'action': 'Emergency portable services',
            'timeline': 'Immediate (24-48 hours)',
            'budget': '₹10-20 lakhs',
            'dept': 'UIDAI + Disaster Management'
        }
    }
    
    def get_recommendation(self, risk_type, district, risk_score):
        """Get detailed policy recommendation"""
        policy = self.POLICY_MATRIX.get(risk_type, {})
        urgency = 'CRITICAL' if risk_score >= 0.8 else 'HIGH' if risk_score >= 0.6 else 'MEDIUM'
        
        return {
            'district': district,
            'risk_type': risk_type,
            'urgency': urgency,
            'action': policy.get('action', 'Monitor situation'),
            'timeline': policy.get('timeline', '1-2 weeks'),
            'budget': policy.get('budget', '₹1-2 lakhs'),
            'department': policy.get('dept', 'UIDAI')
        }
    
    def generate_report(self, results):
        """Generate comprehensive policy report"""
        high_priority = results[results['Alert'] == 'Yes']
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_districts': len(results),
            'high_risk': len(high_priority),
            'recommendations': []
        }
        
        for _, row in high_priority.iterrows():
            rec = self.get_recommendation(row['Risk Type'], row['District'], row['Risk Score'])
            report['recommendations'].append(rec)
        
        return report
