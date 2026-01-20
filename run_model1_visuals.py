"""
Model 1: Anomaly Detection Visualization Script
Generates all required visualizations for the Aadhaar anomaly detection system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from adhaar_anomaly_detector import AadhaarAnomalyDetector

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the cleaned Aadhaar data"""
    try:
        bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
        demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')

        print(f"✓ Loaded biometric data: {len(bio_df)} records")
        print(f"✓ Loaded demographic data: {len(demo_df)} records")

        return bio_df, demo_df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def create_anomaly_visualizations():
    """Generate all Model 1 anomaly detection visualizations"""

    # Load data
    bio_df, demo_df = load_data()
    if bio_df is None or demo_df is None:
        return

    # Initialize detector
    detector = AadhaarAnomalyDetector()

    # Run analysis
    print("\n🔍 Running anomaly detection analysis...")
    final_df, report_df = detector.run_full_analysis(demo_df, bio_df)

    # Create output directory
    output_dir = "model1_anomaly_visuals"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📊 Generating visualizations in {output_dir}/")

    # Visualization 1: Total Enrolment Anomalies
    plt.figure(figsize=(15, 8))
    sample_districts = final_df['District'].unique()[:5]  # Show first 5 districts

    for district in sample_districts:
        district_data = final_df[final_df['District'] == district].copy()
        district_data['Month'] = pd.to_datetime(district_data['Month'])

        plt.plot(district_data['Month'], district_data['Enrolments'],
                label=f'{district} (Actual)', alpha=0.7)
        plt.plot(district_data['Month'], district_data['Enrolments_Mean'],
                label=f'{district} (Expected)', linestyle='--', alpha=0.7)

        # Highlight anomalies
        anomalies = district_data[district_data['Enrolments_ConfirmedAnomaly']]
        if len(anomalies) > 0:
            plt.scatter(anomalies['Month'], anomalies['Enrolments'],
                       color='red', s=50, zorder=5, label=f'{district} Anomalies')

    plt.title('Total Enrolment Anomalies - Sample Districts', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Enrolment Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/anomaly_detection_total_enrol.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 2: Biometric Update Anomalies
    plt.figure(figsize=(15, 8))

    for district in sample_districts:
        district_data = final_df[final_df['District'] == district].copy()
        district_data['Month'] = pd.to_datetime(district_data['Month'])

        plt.plot(district_data['Month'], district_data['Updates'],
                label=f'{district} (Actual)', alpha=0.7)
        plt.plot(district_data['Month'], district_data['Updates_Mean'],
                label=f'{district} (Expected)', linestyle='--', alpha=0.7)

        # Highlight anomalies
        anomalies = district_data[district_data['Updates_ConfirmedAnomaly']]
        if len(anomalies) > 0:
            plt.scatter(anomalies['Month'], anomalies['Updates'],
                       color='red', s=50, zorder=5, label=f'{district} Anomalies')

    plt.title('Biometric Update Anomalies - Sample Districts', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Update Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/anomaly_detection_bio_updates.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 3: Child Enrolment Anomalies
    plt.figure(figsize=(15, 8))

    for district in sample_districts:
        district_data = final_df[final_df['District'] == district].copy()
        district_data['Month'] = pd.to_datetime(district_data['Month'])

        plt.plot(district_data['Month'], district_data['Child_0_5'],
                label=f'{district} (Actual)', alpha=0.7)
        plt.plot(district_data['Month'], district_data['Child_0_5_Mean'],
                label=f'{district} (Expected)', linestyle='--', alpha=0.7)

        # Highlight anomalies
        anomalies = district_data[district_data['Child_0_5_ConfirmedAnomaly']]
        if len(anomalies) > 0:
            plt.scatter(anomalies['Month'], anomalies['Child_0_5'],
                       color='red', s=50, zorder=5, label=f'{district} Anomalies')

    plt.title('Child Enrolment Anomalies (0-5 years) - Sample Districts', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Child Enrolment Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/anomaly_detection_child_enrol.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 4: Female Enrolment Anomalies
    plt.figure(figsize=(15, 8))

    for district in sample_districts:
        district_data = final_df[final_df['District'] == district].copy()
        district_data['Month'] = pd.to_datetime(district_data['Month'])

        plt.plot(district_data['Month'], district_data['Female'],
                label=f'{district} (Actual)', alpha=0.7)
        plt.plot(district_data['Month'], district_data['Female_Mean'],
                label=f'{district} (Expected)', linestyle='--', alpha=0.7)

        # Highlight anomalies
        anomalies = district_data[district_data['Female_ConfirmedAnomaly']]
        if len(anomalies) > 0:
            plt.scatter(anomalies['Month'], anomalies['Female'],
                       color='red', s=50, zorder=5, label=f'{district} Anomalies')

    plt.title('Female Enrolment Anomalies - Sample Districts', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Female Enrolment Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/anomaly_detection_female_enrol.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 5: Multi-Signal Anomaly Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Multi-Signal Anomaly Detection Dashboard', fontsize=16, fontweight='bold')

    signals = ['Enrolments', 'Updates', 'Child_0_5', 'Female']
    signal_names = ['Total Enrolments', 'Biometric Updates', 'Child Enrolments (0-5)', 'Female Enrolments']
    colors = ['blue', 'green', 'orange', 'red']

    for i, (signal, name, color) in enumerate(zip(signals, signal_names, colors)):
        ax = axes[i//2, i%2]

        # Aggregate anomalies by month
        monthly_anomalies = final_df.groupby('Month')[f'{signal}_ConfirmedAnomaly'].sum()

        # Plot
        monthly_anomalies.plot(kind='bar', ax=ax, color=color, alpha=0.7)
        ax.set_title(f'{name} Anomalies by Month', fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Anomalies')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for j, v in enumerate(monthly_anomalies):
            if v > 0:
                ax.text(j, v + 0.1, str(int(v)), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/anomaly_detection_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 6: Anomaly Summary Statistics
    if len(report_df) > 0:
        summary_stats = detector.get_summary_statistics(report_df)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Anomaly Detection Summary Statistics', fontsize=16, fontweight='bold')

        # Severity breakdown
        severity_data = summary_stats['severity_breakdown']
        if severity_data:
            axes[0, 0].pie(severity_data.values(), labels=severity_data.keys(),
                          autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Anomaly Severity Distribution')

        # Signal breakdown
        signal_data = summary_stats['signal_breakdown']
        if signal_data:
            axes[0, 1].bar(signal_data.keys(), signal_data.values(), color='skyblue')
            axes[0, 1].set_title('Anomalies by Signal Type')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Anomaly type breakdown
        type_data = summary_stats['anomaly_type_breakdown']
        if type_data:
            axes[1, 0].pie(type_data.values(), labels=type_data.keys(),
                          autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Anomaly Type Distribution')

        # Key metrics
        axes[1, 1].axis('off')
        metrics_text = f"""
        Key Metrics:

        Total Anomalies: {summary_stats['total_anomalies']}
        Affected Districts: {summary_stats['affected_districts']}
        ML Validated: {summary_stats['ml_validated_count']}

        Detection Parameters:
        • Rolling Window: {detector.rolling_window} months
        • Z-Threshold: ±{detector.z_threshold}
        • Consecutive Months: {detector.consecutive_months}
        """
        axes[1, 1].text(0.1, 0.8, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/anomaly_detection_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("\n✅ All visualizations generated successfully!")
    print(f"📁 Files saved in: {output_dir}/")
    print("📊 Generated 6 visualization files:")
    print("   - anomaly_detection_total_enrol.png")
    print("   - anomaly_detection_bio_updates.png")
    print("   - anomaly_detection_child_enrol.png")
    print("   - anomaly_detection_female_enrol.png")
    print("   - anomaly_detection_dashboard.png")
    print("   - anomaly_detection_summary.png")

if __name__ == "__main__":
    create_anomaly_visualizations()
