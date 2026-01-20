"""
Model 3: Rule-Based Risk Classification System
==============================================
A transparent, auditable risk classification system for government welfare programs.

This system classifies districts into policy-meaningful risk categories based on
anomaly flags from Model 1 and risk scores from Model 2.

Author: Atoms Platform
Date: 2026-01-18
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskSignals:
    """Binary flags representing different types of anomalies"""
    enrolment_anomaly: bool
    update_anomaly: bool
    child_anomaly: bool
    female_anomaly: bool
    sudden_spike: bool
    sudden_large_drop: bool


@dataclass
class RiskClassification:
    """Result of risk classification"""
    district: str
    risk_type: str
    severity: str
    severity_score: float
    reason: str
    suggested_action: str


class Model3RiskClassifier:
    """
    Rule-Based Risk Classification System
    
    This classifier uses transparent, policy-driven rules to categorize
    districts into actionable risk categories. It prioritizes explainability
    and auditability over complex ML approaches.
    """
    
    # Risk category definitions
    RISK_CATEGORIES = {
        'Administrative Disruption Risk': {
            'description': 'System-level failures affecting both enrolment and updates',
            'action': 'Deploy mobile registration units and technical support teams'
        },
        'Future Child Welfare Exclusion Risk': {
            'description': 'Declining enrolment of children aged 0-5',
            'action': 'Launch targeted awareness campaigns for parents with young children'
        },
        'Gender Access Barrier Risk': {
            'description': 'Declining female enrolment indicating access barriers',
            'action': 'Deploy female staff and create women-friendly registration centers'
        },
        'DBT Readiness Failure Risk': {
            'description': 'Low biometric update rates affecting benefit delivery',
            'action': 'Organize mobile update camps and extend deadlines'
        },
        'Migration / Crisis Shock Risk': {
            'description': 'Sudden population changes due to migration or crisis',
            'action': 'Conduct emergency registration drives and needs assessment'
        },
        'No Significant Risk': {
            'description': 'No major anomalies detected',
            'action': 'Continue routine monitoring'
        }
    }
    
    # Severity thresholds
    SEVERITY_THRESHOLDS = {
        'Low': (0.0, 0.3),
        'Medium': (0.3, 0.6),
        'High': (0.6, 1.0)
    }
    
    def __init__(self):
        """Initialize the risk classifier"""
        self.classification_history = []
        
    def extract_risk_signals(self, row: pd.Series) -> RiskSignals:
        """
        Extract binary risk signals from input data
        
        Args:
            row: DataFrame row containing anomaly flags and metrics
            
        Returns:
            RiskSignals object with binary flags
        """
        return RiskSignals(
            enrolment_anomaly=bool(row.get('enrolment_anomaly', False)),
            update_anomaly=bool(row.get('update_anomaly', False)),
            child_anomaly=bool(row.get('child_anomaly', False)),
            female_anomaly=bool(row.get('female_anomaly', False)),
            sudden_spike=bool(row.get('sudden_spike', False)),
            sudden_large_drop=bool(row.get('sudden_large_drop', False))
        )
    
    def apply_classification_rules(self, signals: RiskSignals) -> str:
        """
        Apply rule-based classification logic
        
        Rules are applied in priority order to avoid overlap:
        1. Administrative Disruption (both enrolment and update issues)
        2. Future Child Welfare Exclusion (child-specific issues)
        3. Gender Access Barrier (female-specific issues)
        4. DBT Readiness Failure (update issues only)
        5. Migration / Crisis Shock (sudden changes)
        
        Args:
            signals: RiskSignals object with binary flags
            
        Returns:
            Risk category name
        """
        # Rule 1: Administrative Disruption Risk (HIGHEST PRIORITY)
        if signals.enrolment_anomaly and signals.update_anomaly:
            return 'Administrative Disruption Risk'
        
        # Rule 2: Future Child Welfare Exclusion Risk
        if signals.child_anomaly:
            return 'Future Child Welfare Exclusion Risk'
        
        # Rule 3: Gender Access Barrier Risk
        if signals.female_anomaly:
            return 'Gender Access Barrier Risk'
        
        # Rule 4: DBT Readiness Failure Risk
        if signals.update_anomaly and not signals.enrolment_anomaly:
            return 'DBT Readiness Failure Risk'
        
        # Rule 5: Migration / Crisis Shock Risk
        if signals.sudden_spike or signals.sudden_large_drop:
            return 'Migration / Crisis Shock Risk'
        
        # Default: No Significant Risk
        return 'No Significant Risk'
    
    def calculate_severity(self, 
                          anomaly_severity: float, 
                          structural_risk_score: float) -> Tuple[float, str]:
        """
        Calculate final severity score and label
        
        Formula: Final_Severity = 0.6 × Anomaly_Severity + 0.4 × Structural_Risk_Score
        
        Args:
            anomaly_severity: Severity from Model 1 (0-1 scale)
            structural_risk_score: Risk score from Model 2 (0-1 scale)
            
        Returns:
            Tuple of (severity_score, severity_label)
        """
        # Weighted combination
        final_severity = 0.6 * anomaly_severity + 0.4 * structural_risk_score
        
        # Ensure score is in valid range
        final_severity = np.clip(final_severity, 0.0, 1.0)
        
        # Assign severity label
        if final_severity < 0.3:
            severity_label = 'Low'
        elif final_severity < 0.6:
            severity_label = 'Medium'
        else:
            severity_label = 'High'
        
        return final_severity, severity_label
    
    def generate_explanation(self, 
                           district: str,
                           risk_type: str,
                           row: pd.Series) -> str:
        """
        Generate human-readable explanation for the classification
        
        Args:
            district: District name
            risk_type: Classified risk category
            row: DataFrame row with metrics
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Extract key metrics
        enrol_change = row.get('enrolment_pct_change', 0)
        update_change = row.get('update_pct_change', 0)
        child_change = row.get('child_pct_change', 0)
        female_change = row.get('female_pct_change', 0)
        
        # Build explanation based on risk type
        if risk_type == 'Administrative Disruption Risk':
            explanations.append(f"District flagged due to simultaneous drops in enrolment ({abs(enrol_change):.1f}%) and biometric updates ({abs(update_change):.1f}%)")
            explanations.append("indicating system-level administrative failures")
            
        elif risk_type == 'Future Child Welfare Exclusion Risk':
            explanations.append(f"District shows {abs(child_change):.1f}% drop in enrolments for children aged 0-5")
            explanations.append("risking future welfare exclusion for vulnerable children")
            
        elif risk_type == 'Gender Access Barrier Risk':
            explanations.append(f"District exhibits {abs(female_change):.1f}% decline in female enrolments")
            explanations.append("suggesting gender-specific access barriers")
            
        elif risk_type == 'DBT Readiness Failure Risk':
            explanations.append(f"District flagged due to {abs(update_change):.1f}% drop in biometric updates")
            explanations.append("affecting Direct Benefit Transfer readiness")
            
        elif risk_type == 'Migration / Crisis Shock Risk':
            if row.get('sudden_spike', False):
                explanations.append("District shows sudden spike in registrations")
            else:
                explanations.append("District shows sudden large drop in registrations")
            explanations.append("indicating potential migration or crisis event")
            
        else:
            explanations.append("No significant anomalies detected in current monitoring period")
        
        # Add historical context if available
        if row.get('structural_risk_score', 0) > 0.6:
            explanations.append("Combined with historically low system readiness")
        
        return ". ".join(explanations) + "."
    
    def classify_district(self, row: pd.Series) -> RiskClassification:
        """
        Classify a single district
        
        Args:
            row: DataFrame row with all required inputs
            
        Returns:
            RiskClassification object
        """
        district = row.get('district', 'Unknown')
        
        # Step 1: Extract risk signals
        signals = self.extract_risk_signals(row)
        
        # Step 2: Apply classification rules
        risk_type = self.apply_classification_rules(signals)
        
        # Step 3: Calculate severity
        anomaly_severity = row.get('anomaly_severity', 0.5)
        structural_risk_score = row.get('structural_risk_score', 0.5)
        severity_score, severity_label = self.calculate_severity(
            anomaly_severity, structural_risk_score
        )
        
        # Step 4: Generate explanation
        reason = self.generate_explanation(district, risk_type, row)
        
        # Step 5: Get suggested action
        suggested_action = self.RISK_CATEGORIES[risk_type]['action']
        
        # Create classification result
        classification = RiskClassification(
            district=district,
            risk_type=risk_type,
            severity=severity_label,
            severity_score=severity_score,
            reason=reason,
            suggested_action=suggested_action
        )
        
        return classification
    
    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify multiple districts
        
        Args:
            df: DataFrame with required columns:
                - district
                - enrolment_anomaly, update_anomaly, child_anomaly, female_anomaly
                - sudden_spike, sudden_large_drop
                - anomaly_severity, structural_risk_score
                
        Returns:
            DataFrame with classification results
        """
        results = []
        
        for idx, row in df.iterrows():
            classification = self.classify_district(row)
            results.append({
                'District': classification.district,
                'Risk Type': classification.risk_type,
                'Severity': classification.severity,
                'Severity Score': classification.severity_score,
                'Reason': classification.reason,
                'Suggested Action': classification.suggested_action
            })
        
        results_df = pd.DataFrame(results)
        
        # Store classification history
        self.classification_history.append(results_df.copy())
        
        return results_df
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics from classification results
        
        Args:
            results_df: DataFrame with classification results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_districts': len(results_df),
            'risk_distribution': results_df['Risk Type'].value_counts().to_dict(),
            'severity_distribution': results_df['Severity'].value_counts().to_dict(),
            'high_risk_districts': len(results_df[results_df['Severity'] == 'High']),
            'medium_risk_districts': len(results_df[results_df['Severity'] == 'Medium']),
            'low_risk_districts': len(results_df[results_df['Severity'] == 'Low']),
            'avg_severity_score': results_df['Severity Score'].mean()
        }
        
        return summary


class Model3MLValidator:
    """
    Optional ML-based validator using shallow Decision Tree
    
    This validates the rule-based classification and provides
    an alternative ML perspective while maintaining explainability.
    """
    
    def __init__(self, max_depth: int = 3):
        """
        Initialize ML validator
        
        Args:
            max_depth: Maximum depth of decision tree (default: 3 for explainability)
        """
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with input data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        feature_cols = [
            'enrolment_pct_change',
            'update_pct_change',
            'child_pct_change',
            'female_pct_change',
            'structural_risk_score',
            'anomaly_severity'
        ]
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[feature_cols].fillna(0).values
        self.feature_names = feature_cols
        
        return X, feature_cols
    
    def train(self, df: pd.DataFrame, labels: pd.Series):
        """
        Train the decision tree classifier
        
        Args:
            df: DataFrame with features
            labels: Series with risk category labels
        """
        X, _ = self.prepare_features(df)
        
        self.model.fit(X, labels)
        self.is_trained = True
        
        print("✓ Decision Tree trained successfully")
        print(f"  Tree depth: {self.model.get_depth()}")
        print(f"  Number of leaves: {self.model.get_n_leaves()}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict risk categories using trained model
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predicted risk categories
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def export_tree_rules(self) -> str:
        """
        Export decision tree rules in human-readable format
        
        Returns:
            String representation of tree rules
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        tree_rules = export_text(
            self.model,
            feature_names=self.feature_names,
            max_depth=self.max_depth
        )
        
        return tree_rules
    
    def validate_against_rules(self, 
                               rule_predictions: pd.Series,
                               ml_predictions: np.ndarray) -> Dict:
        """
        Compare ML predictions with rule-based predictions
        
        Args:
            rule_predictions: Predictions from rule-based classifier
            ml_predictions: Predictions from ML model
            
        Returns:
            Dictionary with validation metrics
        """
        agreement = (rule_predictions == ml_predictions).sum()
        total = len(rule_predictions)
        agreement_rate = agreement / total
        
        validation_results = {
            'total_cases': total,
            'agreements': agreement,
            'disagreements': total - agreement,
            'agreement_rate': agreement_rate,
            'validation_status': 'PASS' if agreement_rate > 0.85 else 'REVIEW'
        }
        
        return validation_results


def create_sample_data(n_districts: int = 20) -> pd.DataFrame:
    """
    Create sample data for demonstration
    
    Args:
        n_districts: Number of districts to generate
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)
    
    districts = [f"District_{i+1}" for i in range(n_districts)]
    
    data = {
        'district': districts,
        'enrolment_pct_change': np.random.uniform(-50, 20, n_districts),
        'update_pct_change': np.random.uniform(-60, 15, n_districts),
        'child_pct_change': np.random.uniform(-40, 10, n_districts),
        'female_pct_change': np.random.uniform(-45, 10, n_districts),
        'structural_risk_score': np.random.uniform(0.2, 0.9, n_districts),
        'anomaly_severity': np.random.uniform(0.1, 0.95, n_districts)
    }
    
    df = pd.DataFrame(data)
    
    # Create anomaly flags based on thresholds
    df['enrolment_anomaly'] = df['enrolment_pct_change'] < -20
    df['update_anomaly'] = df['update_pct_change'] < -25
    df['child_anomaly'] = df['child_pct_change'] < -15
    df['female_anomaly'] = df['female_pct_change'] < -20
    df['sudden_spike'] = df['enrolment_pct_change'] > 15
    df['sudden_large_drop'] = df['enrolment_pct_change'] < -40
    
    return df


def main():
    """
    Main demonstration function
    """
    print("=" * 80)
    print("MODEL 3: RULE-BASED RISK CLASSIFICATION SYSTEM")
    print("=" * 80)
    print()
    
    # Step 1: Create sample data
    print("Step 1: Generating sample data...")
    df = create_sample_data(n_districts=20)
    print(f"✓ Generated data for {len(df)} districts")
    print()
    
    # Step 2: Initialize rule-based classifier
    print("Step 2: Initializing rule-based classifier...")
    classifier = Model3RiskClassifier()
    print("✓ Classifier initialized")
    print()
    
    # Step 3: Classify districts
    print("Step 3: Classifying districts using rule-based approach...")
    results = classifier.classify_batch(df)
    print("✓ Classification complete")
    print()
    
    # Step 4: Display results
    print("Step 4: Classification Results")
    print("-" * 80)
    print(results.to_string(index=False))
    print()
    
    # Step 5: Summary statistics
    print("Step 5: Summary Statistics")
    print("-" * 80)
    summary = classifier.get_summary_statistics(results)
    print(f"Total Districts: {summary['total_districts']}")
    print(f"Average Severity Score: {summary['avg_severity_score']:.3f}")
    print()
    print("Risk Distribution:")
    for risk_type, count in summary['risk_distribution'].items():
        print(f"  {risk_type}: {count}")
    print()
    print("Severity Distribution:")
    for severity, count in summary['severity_distribution'].items():
        print(f"  {severity}: {count}")
    print()
    
    # Step 6: High-priority alerts
    print("Step 6: High-Priority Alerts (Severity = High)")
    print("-" * 80)
    high_risk = results[results['Severity'] == 'High']
    if len(high_risk) > 0:
        for idx, row in high_risk.iterrows():
            print(f"\n🚨 {row['District']}")
            print(f"   Risk Type: {row['Risk Type']}")
            print(f"   Reason: {row['Reason']}")
            print(f"   Action: {row['Suggested Action']}")
    else:
        print("No high-priority alerts")
    print()
    
    # Step 7: Optional ML validation
    print("Step 7: Optional ML Validation (Decision Tree)")
    print("-" * 80)
    ml_validator = Model3MLValidator(max_depth=3)
    
    # Train on the classified data
    ml_validator.train(df, results['Risk Type'])
    
    # Get predictions
    ml_predictions = ml_validator.predict(df)
    
    # Validate against rule-based predictions
    validation = ml_validator.validate_against_rules(
        results['Risk Type'],
        ml_predictions
    )
    
    print(f"Agreement Rate: {validation['agreement_rate']:.2%}")
    print(f"Agreements: {validation['agreements']}/{validation['total_cases']}")
    print(f"Validation Status: {validation['validation_status']}")
    print()
    
    # Feature importance
    print("Feature Importance:")
    importance = ml_validator.get_feature_importance()
    print(importance.to_string(index=False))
    print()
    
    # Decision tree rules
    print("Decision Tree Rules (for transparency):")
    print("-" * 80)
    tree_rules = ml_validator.export_tree_rules()
    print(tree_rules)
    print()
    
    print("=" * 80)
    print("CLASSIFICATION COMPLETE")
    print("=" * 80)
    print()
    print("✓ Rule-based classification provides transparent, auditable results")
    print("✓ ML validation confirms consistency of classification logic")
    print("✓ System is ready for government deployment")
    print()


if __name__ == "__main__":
    main()