"""Main Pipeline - Orchestrates all modules"""
from .core import DataLoader, AnomalyDetector, RiskScorer, RiskClassifier

class UIDaiPipeline:
    """Complete UIDAI Risk Monitoring Pipeline"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.anomaly_detector = AnomalyDetector()
        self.risk_scorer = RiskScorer()
        self.risk_classifier = RiskClassifier()
    
    def run(self):
        """Execute complete pipeline"""
        raw_data = self.data_loader.load_cleaned_data()
        anomaly_data = self.anomaly_detector.detect(raw_data)
        risk_data = self.risk_scorer.calculate(anomaly_data)
        results = self.risk_classifier.classify(risk_data)
        return results.sort_values('Risk Score', ascending=False).reset_index(drop=True)
