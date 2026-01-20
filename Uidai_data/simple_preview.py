#!/usr/bin/env python3
"""
Financial Inclusion Scout - Simple Preview
Using actual cleaned data from analysis2.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_cleaned_data():
    """Load and process the actual demographic data"""
    print("Loading actual UIDAI demographic data...")
    
    # Load the datasets
    files = [
        "api_data_aadhar_demographic_0_500000.csv",
        "api_data_aadhar_demographic_500000_1000000.csv", 
        "api_data_aadhar_demographic_1000000_1500000.csv",
        "api_data_aadhar_demographic_1500000_2000000.csv",
        "api_data_aadhar_demographic_2000000_2071700.csv"
    ]
    
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df['new_date'] = pd.to_datetime(df['date'], format='%d-%m-%Y').dt.strftime('%Y%m%d')
            dfs.append(df)
        except:
            print(f"Could not load {file}")
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        
        # Basic cleaning
        df = df.dropna(subset=['state', 'district', 'pincode'])
        df = df[df['pincode'].astype(str).str.match(r'^\d{6}$')]
        df['total_enroll'] = df['demo_age_5_17'] + df['demo_age_17_']
        df['year'] = pd.to_datetime(df['new_date'], format='%Y%m%d').dt.year
        df['month'] = pd.to_datetime(df['new_date'], format='%Y%m%d').dt.month
        
        print(f"Loaded {len(df)} records")
        return df
    return None

def calculate_simple_firs(df):
    """Calculate a simple FIRS score using actual enrollment data"""
    
    # Aggregate by state and district
    state_district = df.groupby(['state', 'district']).agg({
        'total_enroll': 'sum',
        'demo_age_5_17': 'sum', 
        'demo_age_17_': 'sum'
    }).reset_index()
    
    # Simple FIRS calculation (adult enrollment rate as proxy for digital readiness)
    state_district['adult_rate'] = state_district['demo_age_17_'] / state_district['total_enroll']
    state_district['firs_score'] = (state_district['adult_rate'] * 100).round(2)
    
    # Classify regions
    high_threshold = state_district['firs_score'].quantile(0.75)
    low_threshold = state_district['firs_score'].quantile(0.25)
    
    def classify(score):
        if score >= high_threshold:
            return 'High Readiness'
        elif score >= low_threshold:
            return 'Medium Readiness'
        else:
            return 'Low Readiness'
    
    state_district['readiness_zone'] = state_district['firs_score'].apply(classify)
    
    return state_district

def show_preview():
    """Show preview of Financial Inclusion Scout results"""
    print("=" * 60)
    print("FINANCIAL INCLUSION SCOUT - PREVIEW")
    print("=" * 60)
    
    # Load data
    df = load_cleaned_data()
    if df is None:
        print("Could not load data files")
        return
    
    # Calculate FIRS scores
    firs_data = calculate_simple_firs(df)
    
    print(f"\nAnalyzed {len(firs_data)} districts across India")
    print(f"Average FIRS Score: {firs_data['firs_score'].mean():.2f}")
    
    # Top 10 ready regions
    print("\nTOP 10 DIGITALLY READY REGIONS:")
    print("-" * 40)
    top_10 = firs_data.nlargest(10, 'firs_score')
    for _, row in top_10.iterrows():
        print(f"{row['state']}, {row['district']}: {row['firs_score']:.1f}")
    
    # Readiness distribution
    print("\nREADINESS DISTRIBUTION:")
    print("-" * 30)
    zone_counts = firs_data['readiness_zone'].value_counts()
    total = len(firs_data)
    for zone, count in zone_counts.items():
        pct = (count/total)*100
        print(f"{zone}: {count} districts ({pct:.1f}%)")
    
    # State-wise summary
    print("\nTOP 10 STATES BY AVERAGE READINESS:")
    print("-" * 40)
    state_avg = firs_data.groupby('state')['firs_score'].mean().nlargest(10)
    for state, score in state_avg.items():
        print(f"{state}: {score:.1f}")
    
    # Sample data
    print("\nSAMPLE DATA:")
    print("-" * 20)
    sample = firs_data.head(5)[['state', 'district', 'total_enroll', 'firs_score', 'readiness_zone']]
    print(sample.to_string(index=False))
    
    return firs_data

def create_simple_chart(firs_data):
    """Create a simple visualization"""
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Financial Inclusion Scout - Key Insights', fontsize=14, fontweight='bold')
    
    # 1. FIRS Score Distribution
    axes[0,0].hist(firs_data['firs_score'], bins=20, alpha=0.7, color='lightblue')
    axes[0,0].set_title('FIRS Score Distribution')
    axes[0,0].set_xlabel('FIRS Score')
    axes[0,0].set_ylabel('Number of Districts')
    
    # 2. Readiness Zones
    zone_counts = firs_data['readiness_zone'].value_counts()
    axes[0,1].pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Readiness Zone Distribution')
    
    # 3. Top 10 States
    state_avg = firs_data.groupby('state')['firs_score'].mean().nlargest(10)
    axes[1,0].barh(range(len(state_avg)), state_avg.values)
    axes[1,0].set_yticks(range(len(state_avg)))
    axes[1,0].set_yticklabels(state_avg.index)
    axes[1,0].set_title('Top 10 States by FIRS Score')
    
    # 4. Enrollment vs Score
    axes[1,1].scatter(firs_data['total_enroll'], firs_data['firs_score'], alpha=0.6)
    axes[1,1].set_title('Total Enrollment vs FIRS Score')
    axes[1,1].set_xlabel('Total Enrollment')
    axes[1,1].set_ylabel('FIRS Score')
    
    plt.tight_layout()
    plt.savefig('firs_preview.png', dpi=150, bbox_inches='tight')
    print(f"\nChart saved as 'firs_preview.png'")
    plt.show()

if __name__ == "__main__":
    firs_data = show_preview()
    if firs_data is not None:
        create_simple_chart(firs_data)