#!/usr/bin/env python3
"""
Financial Inclusion Scout - Preview Dashboard
Quick preview of key insights and visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from financial_inclusion_scout import FinancialInclusionScout

def create_preview():
    """Generate preview of Financial Inclusion Scout results"""
    print("FINANCIAL INCLUSION SCOUT - PREVIEW DASHBOARD")
    print("=" * 60)
    
    # Run analysis
    scout = FinancialInclusionScout()
    results = scout.run_complete_analysis()
    
    # Get data
    firs_data = results['firs_scores']
    
    print("\n" + "=" * 60)
    print("KEY PREVIEW INSIGHTS")
    print("=" * 60)
    
    # 1. Top 10 Ready Regions
    print("\n1. TOP 10 DIGITALLY READY REGIONS:")
    top_10 = firs_data.nlargest(10, 'firs_score')[['state_clean', 'district', 'firs_score']]
    for i, row in top_10.iterrows():
        print(f"   {row['state_clean'].title()}, {row['district']} - Score: {row['firs_score']}")
    
    # 2. State-wise Summary
    print("\n2. STATE-WISE READINESS SUMMARY:")
    state_summary = (firs_data.groupby(['state_clean', 'readiness_zone'])
                    .size().unstack(fill_value=0))
    print(state_summary.head(10))
    
    # 3. Distribution Analysis
    print("\n3. FIRS SCORE DISTRIBUTION:")
    print(f"   Mean Score: {firs_data['firs_score'].mean():.2f}")
    print(f"   Median Score: {firs_data['firs_score'].median():.2f}")
    print(f"   Standard Deviation: {firs_data['firs_score'].std():.2f}")
    
    # 4. Readiness Zone Breakdown
    print("\n4. NATIONAL READINESS BREAKDOWN:")
    zone_counts = firs_data['readiness_zone'].value_counts()
    total = len(firs_data)
    for zone, count in zone_counts.items():
        pct = (count/total)*100
        print(f"   {zone}: {count} regions ({pct:.1f}%)")
    
    # 5. Create visualizations
    create_visualizations(firs_data)
    
    print("\n" + "=" * 60)
    print("PREVIEW COMPLETE - Check generated charts!")
    print("=" * 60)

def create_visualizations(data):
    """Create key visualizations"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Financial Inclusion Scout - Key Insights', fontsize=16, fontweight='bold')
    
    # 1. FIRS Score Distribution
    axes[0,0].hist(data['firs_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('FIRS Score Distribution')
    axes[0,0].set_xlabel('FIRS Score')
    axes[0,0].set_ylabel('Number of Regions')
    axes[0,0].axvline(data['firs_score'].mean(), color='red', linestyle='--', label=f'Mean: {data["firs_score"].mean():.1f}')
    axes[0,0].legend()
    
    # 2. Readiness Zone Pie Chart
    zone_counts = data['readiness_zone'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[0,1].pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0,1].set_title('Regional Readiness Distribution')
    
    # 3. Top 10 States by Average FIRS Score
    state_avg = data.groupby('state_clean')['firs_score'].mean().nlargest(10)
    axes[1,0].barh(range(len(state_avg)), state_avg.values, color='lightgreen')
    axes[1,0].set_yticks(range(len(state_avg)))
    axes[1,0].set_yticklabels([s.title() for s in state_avg.index])
    axes[1,0].set_title('Top 10 States by Avg FIRS Score')
    axes[1,0].set_xlabel('Average FIRS Score')
    
    # 4. Enrollment vs FIRS Score Scatter
    sample_data = data.sample(min(500, len(data)))  # Sample for readability
    scatter = axes[1,1].scatter(sample_data['base_population'], sample_data['firs_score'], 
                               c=sample_data['firs_score'], cmap='viridis', alpha=0.6)
    axes[1,1].set_title('Enrollment vs FIRS Score')
    axes[1,1].set_xlabel('Base Population (Enrollments)')
    axes[1,1].set_ylabel('FIRS Score')
    plt.colorbar(scatter, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('financial_inclusion_scout_preview.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'financial_inclusion_scout_preview.png'")
    plt.show()

def show_sample_data():
    """Show sample of the processed data"""
    scout = FinancialInclusionScout()
    scout.load_and_clean_data()
    scout.extract_core_signals()
    scout.calculate_firs_score()
    scout.classify_regions()
    
    print("\nSAMPLE DATA PREVIEW:")
    print("-" * 50)
    sample = scout.firs_scores.head(10)[['state_clean', 'district', 'year', 'base_population', 'mobile_updates', 'firs_score', 'readiness_zone']]
    print(sample.to_string(index=False))

if __name__ == "__main__":
    create_preview()
    show_sample_data()