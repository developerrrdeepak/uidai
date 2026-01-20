#!/usr/bin/env python3
"""
Financial Inclusion Scout - Working Version
Handles actual UIDAI data columns correctly
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("FINANCIAL INCLUSION SCOUT - AADHAAR SERVICE LOAD FORECASTING")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
enrol = pd.read_csv('api_data_aadhar_enrolment_0_500000.csv')
demo = pd.read_csv('api_data_aadhar_demographic_0_500000.csv')
bio = pd.read_csv('api_data_aadhar_biometric_0_500000.csv')
print(f"[OK] Loaded: Enrolment ({len(enrol):,}), Demographic ({len(demo):,}), Biometric ({len(bio):,})")

# Clean data
print("\n[2/5] Cleaning data...")

def clean_df(df, name):
    df.columns = df.columns.str.lower().str.strip()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['state'] = df['state'].astype(str).str.lower().str.strip()
    df['district'] = df['district'].astype(str).str.lower().str.strip()
    df['year_month'] = df['date'].dt.to_period('M')
    df = df.drop_duplicates(subset=['date', 'state', 'district'])
    print(f"  {name}: {len(df):,} records")
    return df

enrol = clean_df(enrol, "Enrolment")
demo = clean_df(demo, "Demographic")
bio = clean_df(bio, "Biometric")

# Aggregate
print("\n[3/5] Aggregating to monthly district level...")

enrol_agg = enrol.groupby(['state', 'district', 'year_month']).agg({
    'age_18_greater': 'sum'
}).reset_index().rename(columns={'age_18_greater': 'enrolment_load'})

demo_agg = demo.groupby(['state', 'district', 'year_month']).agg({
    'demo_age_17_': 'sum'
}).reset_index().rename(columns={'demo_age_17_': 'demographic_load'})

bio_agg = bio.groupby(['state', 'district', 'year_month']).agg({
    'bio_age_17_': 'sum'
}).reset_index().rename(columns={'bio_age_17_': 'biometric_load'})

print(f"[OK] Aggregated: {len(enrol_agg):,} enrolment, {len(demo_agg):,} demographic, {len(bio_agg):,} biometric")

# Merge and compute service load
print("\n[4/5] Computing service load and metrics...")

merged = enrol_agg.merge(demo_agg, on=['state', 'district', 'year_month'], how='outer')
merged = merged.merge(bio_agg, on=['state', 'district', 'year_month'], how='outer')
merged = merged.fillna(0)

merged['total_service_load'] = (
    merged['enrolment_load'] * 1.0 +
    merged['demographic_load'] * 0.6 +
    merged['biometric_load'] * 0.8
)

merged['firs'] = (
    0.5 * (merged['enrolment_load'] / (merged['total_service_load'] + 1)) +
    0.3 * (merged['biometric_load'] / (merged['total_service_load'] + 1)) +
    0.2
)

# Compute trends
merged = merged.sort_values(['state', 'district', 'year_month'])
merged['ma_6m'] = merged.groupby(['state', 'district'])['total_service_load'].transform(
    lambda x: x.rolling(6, min_periods=1).mean()
)

print(f"[OK] Computed metrics for {len(merged):,} district-month combinations")

# Generate priorities
print("\n[5/5] Generating priorities and recommendations...")

latest = merged.groupby(['state', 'district']).tail(1).copy()

# Forecast (simple: use 6-month average)
latest['forecasted_load'] = latest['ma_6m']

# Priority score
latest['norm_forecast'] = (latest['forecasted_load'] - latest['forecasted_load'].min()) / \
                          (latest['forecasted_load'].max() - latest['forecasted_load'].min() + 1)
latest['norm_firs'] = 1 - latest['firs']
latest['service_gap'] = latest['total_service_load'] / (latest['ma_6m'] + 1)
latest['norm_gap'] = (latest['service_gap'] - latest['service_gap'].min()) / \
                     (latest['service_gap'].max() - latest['service_gap'].min() + 1)

latest['priority_score'] = (
    latest['norm_forecast'] * 0.4 +
    latest['norm_firs'] * 0.3 +
    latest['norm_gap'] * 0.3
)

latest = latest.sort_values('priority_score', ascending=False)
latest['rank'] = range(1, len(latest) + 1)

n = len(latest)
latest['priority_tier'] = pd.cut(
    latest['rank'],
    bins=[0, n*0.1, n*0.25, n*0.5, n],
    labels=['Critical', 'High', 'Medium', 'Low']
)

print(f"[OK] Ranked {len(latest):,} districts")

# Save outputs
merged.to_csv('output_service_load.csv', index=False)
latest.to_csv('output_priority_rankings.csv', index=False)
latest.head(20).to_csv('output_camp_recommendations.csv', index=False)

print("\n" + "="*70)
print("OUTPUTS SAVED")
print("="*70)
print("[OK] output_service_load.csv")
print("[OK] output_priority_rankings.csv")
print("[OK] output_camp_recommendations.csv")

# Display results
print("\n" + "="*70)
print("TOP 10 PRIORITY DISTRICTS FOR CAMP PLACEMENT")
print("="*70)

top10 = latest.head(10)[['state', 'district', 'priority_score', 'priority_tier', 
                          'forecasted_load', 'firs', 'total_service_load']]
print(top10.to_string(index=False))

print("\n" + "="*70)
print("PRIORITY TIER DISTRIBUTION")
print("="*70)
print(latest['priority_tier'].value_counts().to_string())

print("\n" + "="*70)
print("ANALYSIS COMPLETE [SUCCESS]")
print("="*70)
