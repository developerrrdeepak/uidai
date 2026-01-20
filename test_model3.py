"""
Test Suite for Model 3 Risk Classification System
==================================================
Comprehensive tests to ensure system reliability
"""

import unittest
import pandas as pd
import numpy as np
from model3_risk_classification import (
    Model3RiskClassifier,
    Model3MLValidator,
    RiskSignals,
    RiskClassification,
    create_sample_data
)


class TestRiskSignals(unittest.TestCase):
    """Test RiskSignals dataclass"""
    
    def test_risk_signals_creation(self):
        """Test creating RiskSignals object"""
        signals = RiskSignals(
            enrolment_anomaly=True,
            update_anomaly=False,
            child_anomaly=True,
            female_anomaly=False,
            sudden_spike=False,
            sudden_large_drop=True
        )
        
        self.assertTrue(signals.enrolment_anomaly)
        self.assertFalse(signals.update_anomaly)
        self.assertTrue(signals.child_anomaly)


class TestModel3RiskClassifier(unittest.TestCase):
    """Test Model3RiskClassifier class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = Model3RiskClassifier()
        
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsInstance(self.classifier, Model3RiskClassifier)
        self.assertEqual(len(self.classifier.classification_history), 0)
        
    def test_extract_risk_signals(self):
        """Test risk signal extraction"""
        row = pd.Series({
            'enrolment_anomaly': True,
            'update_anomaly': False,
            'child_anomaly': True,
            'female_anomaly': False,
            'sudden_spike': False,
            'sudden_large_drop': False
        })
        
        signals = self.classifier.extract_risk_signals(row)
        
        self.assertTrue(signals.enrolment_anomaly)
        self.assertFalse(signals.update_anomaly)
        self.assertTrue(signals.child_anomaly)
        
    def test_classification_rule_1_administrative_disruption(self):
        """Test Rule 1: Administrative Disruption Risk"""
        signals = RiskSignals(
            enrolment_anomaly=True,
            update_anomaly=True,
            child_anomaly=False,
            female_anomaly=False,
            sudden_spike=False,
            sudden_large_drop=False
        )
        
        risk_type = self.classifier.apply_classification_rules(signals)
        
        self.assertEqual(risk_type, 'Administrative Disruption Risk')
        
    def test_classification_rule_2_child_welfare(self):
        """Test Rule 2: Future Child Welfare Exclusion Risk"""
        signals = RiskSignals(
            enrolment_anomaly=False,
            update_anomaly=False,
            child_anomaly=True,
            female_anomaly=False,
            sudden_spike=False,
            sudden_large_drop=False
        )
        
        risk_type = self.classifier.apply_classification_rules(signals)
        
        self.assertEqual(risk_type, 'Future Child Welfare Exclusion Risk')
        
    def test_classification_rule_3_gender_barrier(self):
        """Test Rule 3: Gender Access Barrier Risk"""
        signals = RiskSignals(
            enrolment_anomaly=False,
            update_anomaly=False,
            child_anomaly=False,
            female_anomaly=True,
            sudden_spike=False,
            sudden_large_drop=False
        )
        
        risk_type = self.classifier.apply_classification_rules(signals)
        
        self.assertEqual(risk_type, 'Gender Access Barrier Risk')
        
    def test_classification_rule_4_dbt_readiness(self):
        """Test Rule 4: DBT Readiness Failure Risk"""
        signals = RiskSignals(
            enrolment_anomaly=False,
            update_anomaly=True,
            child_anomaly=False,
            female_anomaly=False,
            sudden_spike=False,
            sudden_large_drop=False
        )
        
        risk_type = self.classifier.apply_classification_rules(signals)
        
        self.assertEqual(risk_type, 'DBT Readiness Failure Risk')
        
    def test_classification_rule_5_migration_crisis(self):
        """Test Rule 5: Migration / Crisis Shock Risk"""
        signals = RiskSignals(
            enrolment_anomaly=False,
            update_anomaly=False,
            child_anomaly=False,
            female_anomaly=False,
            sudden_spike=True,
            sudden_large_drop=False
        )
        
        risk_type = self.classifier.apply_classification_rules(signals)
        
        self.assertEqual(risk_type, 'Migration / Crisis Shock Risk')
        
    def test_classification_no_risk(self):
        """Test no significant risk classification"""
        signals = RiskSignals(
            enrolment_anomaly=False,
            update_anomaly=False,
            child_anomaly=False,
            female_anomaly=False,
            sudden_spike=False,
            sudden_large_drop=False
        )
        
        risk_type = self.classifier.apply_classification_rules(signals)
        
        self.assertEqual(risk_type, 'No Significant Risk')
        
    def test_severity_calculation_low(self):
        """Test severity calculation - Low"""
        score, label = self.classifier.calculate_severity(0.2, 0.2)
        
        self.assertLess(score, 0.3)
        self.assertEqual(label, 'Low')
        
    def test_severity_calculation_medium(self):
        """Test severity calculation - Medium"""
        score, label = self.classifier.calculate_severity(0.5, 0.5)
        
        self.assertGreaterEqual(score, 0.3)
        self.assertLess(score, 0.6)
        self.assertEqual(label, 'Medium')
        
    def test_severity_calculation_high(self):
        """Test severity calculation - High"""
        score, label = self.classifier.calculate_severity(0.8, 0.8)
        
        self.assertGreaterEqual(score, 0.6)
        self.assertEqual(label, 'High')
        
    def test_severity_calculation_weights(self):
        """Test severity calculation weights (0.6 and 0.4)"""
        anomaly_sev = 0.5
        structural_risk = 0.5
        
        expected = 0.6 * anomaly_sev + 0.4 * structural_risk
        score, _ = self.classifier.calculate_severity(anomaly_sev, structural_risk)
        
        self.assertAlmostEqual(score, expected, places=5)
        
    def test_classify_district(self):
        """Test single district classification"""
        row = pd.Series({
            'district': 'Test_District',
            'enrolment_anomaly': True,
            'update_anomaly': True,
            'child_anomaly': False,
            'female_anomaly': False,
            'sudden_spike': False,
            'sudden_large_drop': False,
            'anomaly_severity': 0.7,
            'structural_risk_score': 0.6,
            'enrolment_pct_change': -35,
            'update_pct_change': -40,
            'child_pct_change': -5,
            'female_pct_change': -10
        })
        
        classification = self.classifier.classify_district(row)
        
        self.assertIsInstance(classification, RiskClassification)
        self.assertEqual(classification.district, 'Test_District')
        self.assertEqual(classification.risk_type, 'Administrative Disruption Risk')
        self.assertEqual(classification.severity, 'High')
        
    def test_classify_batch(self):
        """Test batch classification"""
        df = create_sample_data(n_districts=10)
        results = self.classifier.classify_batch(df)
        
        self.assertEqual(len(results), 10)
        self.assertIn('District', results.columns)
        self.assertIn('Risk Type', results.columns)
        self.assertIn('Severity', results.columns)
        
    def test_summary_statistics(self):
        """Test summary statistics generation"""
        df = create_sample_data(n_districts=20)
        results = self.classifier.classify_batch(df)
        summary = self.classifier.get_summary_statistics(results)
        
        self.assertEqual(summary['total_districts'], 20)
        self.assertIn('risk_distribution', summary)
        self.assertIn('severity_distribution', summary)
        self.assertIn('avg_severity_score', summary)


class TestModel3MLValidator(unittest.TestCase):
    """Test Model3MLValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = Model3MLValidator(max_depth=3)
        
    def test_initialization(self):
        """Test validator initialization"""
        self.assertIsInstance(self.validator, Model3MLValidator)
        self.assertEqual(self.validator.max_depth, 3)
        self.assertFalse(self.validator.is_trained)
        
    def test_prepare_features(self):
        """Test feature preparation"""
        df = create_sample_data(n_districts=5)
        X, feature_names = self.validator.prepare_features(df)
        
        self.assertEqual(X.shape[0], 5)
        self.assertEqual(len(feature_names), 6)
        self.assertIn('enrolment_pct_change', feature_names)
        
    def test_train_and_predict(self):
        """Test training and prediction"""
        # Generate data
        df = create_sample_data(n_districts=30)
        
        # Get labels from rule-based classifier
        classifier = Model3RiskClassifier()
        results = classifier.classify_batch(df)
        labels = results['Risk Type']
        
        # Train validator
        self.validator.train(df, labels)
        
        self.assertTrue(self.validator.is_trained)
        
        # Predict
        predictions = self.validator.predict(df)
        
        self.assertEqual(len(predictions), 30)
        
    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Generate and train
        df = create_sample_data(n_districts=30)
        classifier = Model3RiskClassifier()
        results = classifier.classify_batch(df)
        
        self.validator.train(df, results['Risk Type'])
        
        # Get importance
        importance = self.validator.get_feature_importance()
        
        self.assertEqual(len(importance), 6)
        self.assertIn('Feature', importance.columns)
        self.assertIn('Importance', importance.columns)
        
    def test_validation_high_agreement(self):
        """Test validation with high agreement"""
        df = create_sample_data(n_districts=50)
        classifier = Model3RiskClassifier()
        results = classifier.classify_batch(df)
        
        self.validator.train(df, results['Risk Type'])
        ml_predictions = self.validator.predict(df)
        
        validation = self.validator.validate_against_rules(
            results['Risk Type'],
            ml_predictions
        )
        
        self.assertIn('agreement_rate', validation)
        self.assertIn('validation_status', validation)
        # Agreement should be reasonably high
        self.assertGreater(validation['agreement_rate'], 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to classification"""
        # Step 1: Generate data
        df = create_sample_data(n_districts=25)
        
        # Step 2: Rule-based classification
        classifier = Model3RiskClassifier()
        results = classifier.classify_batch(df)
        
        # Step 3: Verify results
        self.assertEqual(len(results), 25)
        
        # Step 4: Get summary
        summary = classifier.get_summary_statistics(results)
        self.assertEqual(summary['total_districts'], 25)
        
        # Step 5: ML validation
        validator = Model3MLValidator(max_depth=3)
        validator.train(df, results['Risk Type'])
        ml_predictions = validator.predict(df)
        
        validation = validator.validate_against_rules(
            results['Risk Type'],
            ml_predictions
        )
        
        # Verify validation completed
        self.assertIn('agreement_rate', validation)
        
    def test_high_priority_filtering(self):
        """Test filtering high-priority cases"""
        df = create_sample_data(n_districts=30)
        classifier = Model3RiskClassifier()
        results = classifier.classify_batch(df)
        
        high_priority = results[results['Severity'] == 'High']
        
        # Verify all high priority cases have severity score > 0.6
        for score in high_priority['Severity Score']:
            self.assertGreaterEqual(score, 0.6)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_missing_columns(self):
        """Test handling of missing columns"""
        df = pd.DataFrame({
            'district': ['Test'],
            'enrolment_anomaly': [True]
            # Missing other columns
        })
        
        classifier = Model3RiskClassifier()
        results = classifier.classify_batch(df)
        
        # Should handle gracefully with defaults
        self.assertEqual(len(results), 1)
        
    def test_extreme_values(self):
        """Test handling of extreme values"""
        row = pd.Series({
            'district': 'Extreme_District',
            'enrolment_anomaly': True,
            'update_anomaly': True,
            'child_anomaly': False,
            'female_anomaly': False,
            'sudden_spike': False,
            'sudden_large_drop': False,
            'anomaly_severity': 1.5,  # Out of range
            'structural_risk_score': -0.5,  # Out of range
            'enrolment_pct_change': -100,
            'update_pct_change': 200,
            'child_pct_change': 0,
            'female_pct_change': 0
        })
        
        classifier = Model3RiskClassifier()
        classification = classifier.classify_district(row)
        
        # Should clip severity to valid range
        self.assertGreaterEqual(classification.severity_score, 0.0)
        self.assertLessEqual(classification.severity_score, 1.0)


def run_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING MODEL 3 TEST SUITE")
    print("=" * 80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRiskSignals))
    suite.addTests(loader.loadTestsFromTestCase(TestModel3RiskClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestModel3MLValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    print("=" * 80 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)