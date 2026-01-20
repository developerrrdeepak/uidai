"""Test Suite for UIDAI Risk Monitoring System"""
import unittest
import pandas as pd
import numpy as np
from src.core import DataLoader, AnomalyDetector, RiskScorer, RiskClassifier
from src.pipeline import UIDaiPipeline

class TestDataLoader(unittest.TestCase):
    def test_sample_data_generation(self):
        loader = DataLoader()
        data = loader._create_sample_data(10)
        self.assertEqual(len(data), 10)
        self.assertIn('district', data.columns)

class TestAnomalyDetector(unittest.TestCase):
    def test_anomaly_detection(self):
        detector = AnomalyDetector()
        data = pd.DataFrame({
            'district': ['Test'],
            'enrolment_pct_change': [50],
            'update_pct_change': [-30],
            'child_pct_change': [-25],
            'female_pct_change': [-20]
        })
        result = detector.detect(data)
        self.assertTrue(result['enrolment_anomaly'].iloc[0])
        self.assertTrue(result['update_anomaly'].iloc[0])

class TestRiskScorer(unittest.TestCase):
    def test_risk_calculation(self):
        scorer = RiskScorer()
        data = pd.DataFrame({
            'district': ['Test'],
            'enrolment_anomaly': [True],
            'update_anomaly': [True],
            'child_anomaly': [False],
            'female_anomaly': [False],
            'enrolment_pct_change': [40],
            'update_pct_change': [-30],
            'child_pct_change': [0],
            'female_pct_change': [0]
        })
        result = scorer.calculate(data)
        self.assertIn('risk_score', result.columns)
        self.assertTrue(0 <= result['risk_score'].iloc[0] <= 1)

class TestPipeline(unittest.TestCase):
    def test_complete_pipeline(self):
        pipeline = UIDaiPipeline()
        results = pipeline.run()
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('District', results.columns)
        self.assertIn('Risk Score', results.columns)
        self.assertIn('Risk Level', results.columns)

if __name__ == '__main__':
    unittest.main()
