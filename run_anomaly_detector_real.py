"""
Run Aadhaar Anomaly Detection on Real Data

This script loads the cleaned Aadhaar enrolment and biometric update data
and runs the anomaly detection system.
"""

import pandas as pd
import numpy as np
import os
from adhaar_anomaly_detector import AadhaarAnomalyDetector

def load_enrolment_data():
    """Load enrolment data from cleaned CSV file."""
    file_path = "aadhaar_demographic_cleaned.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned enrolment data file not found: {file_path}")

    enrolment_df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(enrolment_df)} cleaned enrolment records from {file_path}")
    return enrolment_df

def load_biometric_update_data():
    """Load biometric update data from cleaned CSV file."""
    file_path = "aadhaar_biometric_cleaned.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned biometric data file not found: {file_path}")

    update_df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(update_df)} cleaned biometric update records from {file_path}")
    return update_df

def main():
    """
    Main function to run anomaly detection on real data.
    """
    print("=" * 70)
    print("AADHAAR ANOMALY DETECTION SYSTEM - REAL DATA ANALYSIS")
    print("=" * 70)

    # Step 1: Load real data
    print("\n[1] Loading real Aadhaar data...")
    try:
        enrolment_df = load_enrolment_data()
        update_df = load_biometric_update_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Step 2: Initialize detector
    print("\n[2] Initializing anomaly detector...")
    detector = AadhaarAnomalyDetector(
        rolling_window=12,
        z_threshold=2.0,
        consecutive_months=2
    )
    print("✓ Detector initialized with:")
    print("  - Rolling window: 12 months")
    print("  - Z-score threshold: ±2.0")
    print("  - Consecutive months for confirmation: 2")

    # Step 3: Run full analysis
    print("\n[3] Running full anomaly detection analysis...")
    print("-" * 70)
    try:
        full_analysis_df, report_df = detector.run_full_analysis(enrolment_df, update_df)
        print("-" * 70)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return

    # Step 4: Display results
    print("\n[4] Analysis Results:")

    if len(report_df) > 0:
        print("\n📊 Anomaly Report (Top 10 records):")
        print(report_df.head(10).to_string(index=False))

        # Get summary statistics
        summary = detector.get_summary_statistics(report_df)

        print("\n📈 Summary Statistics:")
        print(f"  Total Anomalies: {summary['total_anomalies']}")
        print(f"  Affected Districts: {summary['affected_districts']}")
        print(f"  ML Validated Anomalies: {summary['ml_validated_count']}")

        print("\n  Severity Breakdown:")
        for severity, count in summary['severity_breakdown'].items():
            print(f"    {severity}: {count}")

        print("\n  Signal Breakdown:")
        for signal, count in summary['signal_breakdown'].items():
            print(f"    {signal}: {count}")

        print("\n  Anomaly Type Breakdown:")
        for atype, count in summary['anomaly_type_breakdown'].items():
            print(f"    {atype}: {count}")

        # Save outputs
        print("\n[5] Saving outputs...")
        os.makedirs('output', exist_ok=True)
        report_df.to_csv('output/real_anomaly_report.csv', index=False)
        full_analysis_df.to_csv('output/real_full_analysis.csv', index=False)
        print("✓ Saved real_anomaly_report.csv")
        print("✓ Saved real_full_analysis.csv")

    else:
        print("\n✓ No anomalies detected in the real data")

    print("\n" + "=" * 70)
    print("REAL DATA ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
