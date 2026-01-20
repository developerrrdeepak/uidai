"""
Error Checking Script for UIDAI System
Checks all modules for import errors and runtime issues
"""

import sys
import traceback

def check_module(module_name, import_statement):
    """Check if a module can be imported"""
    try:
        exec(import_statement)
        print(f"[OK] {module_name}: OK")
        return True
    except Exception as e:
        print(f"[ERROR] {module_name}: ERROR")
        print(f"   {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("UIDAI SYSTEM ERROR CHECK")
    print("=" * 80 + "\n")
    
    errors = []
    
    # Check core modules
    print("Checking Core Modules:")
    print("-" * 80)
    if not check_module("DataLoader", "from src.core import DataLoader"):
        errors.append("DataLoader")
    if not check_module("AnomalyDetector", "from src.core import AnomalyDetector"):
        errors.append("AnomalyDetector")
    if not check_module("RiskScorer", "from src.core import RiskScorer"):
        errors.append("RiskScorer")
    if not check_module("RiskClassifier", "from src.core import RiskClassifier"):
        errors.append("RiskClassifier")
    
    print("\nChecking Utils Modules:")
    print("-" * 80)
    if not check_module("MapDataGenerator", "from src.utils import MapDataGenerator"):
        errors.append("MapDataGenerator")
    if not check_module("TrendDataGenerator", "from src.utils import TrendDataGenerator"):
        errors.append("TrendDataGenerator")
    if not check_module("PolicyEngine", "from src.utils import PolicyEngine"):
        errors.append("PolicyEngine")
    if not check_module("ReportExporter", "from src.utils import ReportExporter"):
        errors.append("ReportExporter")
    
    print("\nChecking Pipeline:")
    print("-" * 80)
    if not check_module("UIDaiPipeline", "from src.pipeline import UIDaiPipeline"):
        errors.append("UIDaiPipeline")
    
    print("\nChecking Complete System:")
    print("-" * 80)
    if not check_module("UIDaiCompleteSystem", "from complete_unified_system import UIDaiCompleteSystem"):
        errors.append("UIDaiCompleteSystem")
    
    print("\nChecking Policy Engine:")
    print("-" * 80)
    if not check_module("PolicyRecommendationEngine", "from policy_recommendations import PolicyRecommendationEngine"):
        errors.append("PolicyRecommendationEngine")
    
    # Test pipeline execution
    print("\n" + "=" * 80)
    print("TESTING PIPELINE EXECUTION")
    print("=" * 80)
    
    try:
        from src.pipeline import UIDaiPipeline
        pipeline = UIDaiPipeline()
        print("[OK] Pipeline initialized")
        
        results = pipeline.run()
        print(f"[OK] Pipeline executed successfully")
        print(f"   Generated {len(results)} results")
        print(f"   Columns: {list(results.columns)}")
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed:")
        print(f"   {str(e)}")
        traceback.print_exc()
        errors.append("Pipeline Execution")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if errors:
        print(f"[ERROR] Found {len(errors)} errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("[OK] All checks passed! System is ready.")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
