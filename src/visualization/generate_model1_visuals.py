"""
Model 1: Anomaly Detection Visualization Generator
Generates comprehensive charts for anomaly detection using cleaned data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("  MODEL 1: ANOMALY DETECTION VISUALIZATIONS")
print("="*70)

# Load cleaned data
print("\n[1/3] Loading cleaned CSV files...")
bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')
print(f"[OK] Biometric: {len(bio_df)} rows, {bio_df['district_clean'].nunique()} districts")
print(f"[OK] Demographic: {len(demo_df)} rows, {demo_df['district_clean'].nunique()} districts")

# Aggregate by district
print("\n[2/3] Processing data...")
bio_agg = bio_df.groupby('district_clean').agg({
    'bio_age_5_17': 'sum',
    'bio_age_17_': 'sum'
}).reset_index()
bio_agg['total_bio'] = bio_agg['bio_age_5_17'] + bio_agg['bio_age_17_']

demo_agg = demo_df.groupby('district_clean').agg({
    'demo_age_5_17': 'sum',
    'demo_age_17_': 'sum'
}).reset_index()
demo_agg['total_demo'] = demo_agg['demo_age_5_17'] + demo_agg['demo_age_17_']

df = pd.merge(bio_agg, demo_agg, on='district_clean', how='outer').fillna(0)
df['coverage_ratio'] = df['total_bio'] / (df['total_demo'] + 1)
df['anomaly_score'] = 1 - df['coverage_ratio']

# Detect anomalies (districts with low coverage)
threshold = df['coverage_ratio'].quantile(0.25)
df['is_anomaly'] = df['coverage_ratio'] < threshold

print(f"[OK] Processed {len(df)} districts")
print(f"[OK] Detected {df['is_anomaly'].sum()} anomalies (coverage < {threshold:.3f})")

# Create output directory
os.makedirs('model1_visuals', exist_ok=True)

print("\n[3/3] Generating visualizations...")

# Chart 1: Coverage Distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['coverage_ratio'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Anomaly Threshold ({threshold:.3f})')
ax.set_xlabel('Biometric Coverage Ratio', fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Districts', fontweight='bold', fontsize=12)
ax.set_title('Model 1: Biometric Coverage Distribution Across Districts', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model1_visuals/1_coverage_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 1_coverage_distribution.png")

# Chart 2: Top 20 Anomalies
anomalies = df[df['is_anomaly']].nlargest(20, 'anomaly_score')
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(range(len(anomalies)), anomalies['anomaly_score'], color='crimson', alpha=0.8)
ax.set_yticks(range(len(anomalies)))
ax.set_yticklabels(anomalies['district_clean'], fontsize=10)
ax.set_xlabel('Anomaly Score (1 - Coverage Ratio)', fontweight='bold', fontsize=12)
ax.set_title('Model 1: Top 20 Districts with Coverage Anomalies', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (idx, row) in enumerate(anomalies.iterrows()):
    ax.text(row['anomaly_score'] + 0.01, i, f"{row['anomaly_score']:.3f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig('model1_visuals/2_top_anomalies.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 2_top_anomalies.png")

# Chart 3: Anomaly vs Normal Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
normal = df[~df['is_anomaly']]
anomaly = df[df['is_anomaly']]

ax1.scatter(normal['total_demo'], normal['total_bio'], alpha=0.5, s=30, label='Normal', color='green')
ax1.scatter(anomaly['total_demo'], anomaly['total_bio'], alpha=0.7, s=50, label='Anomaly', color='red', marker='x')
ax1.plot([0, df['total_demo'].max()], [0, df['total_demo'].max()], 'k--', alpha=0.3, label='Perfect Coverage')
ax1.set_xlabel('Demographic Coverage', fontweight='bold', fontsize=12)
ax1.set_ylabel('Biometric Coverage', fontweight='bold', fontsize=12)
ax1.set_title('Biometric vs Demographic Coverage', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

data = [normal['coverage_ratio'], anomaly['coverage_ratio']]
bp = ax2.boxplot(data, labels=['Normal', 'Anomaly'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.set_ylabel('Coverage Ratio', fontweight='bold', fontsize=12)
ax2.set_title('Coverage Ratio Distribution', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model1_visuals/3_anomaly_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 3_anomaly_comparison.png")

# Chart 4: Child vs Adult Coverage
fig, ax = plt.subplots(figsize=(12, 8))
df['child_coverage'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
df['adult_coverage'] = df['bio_age_17_'] / (df['demo_age_17_'] + 1)

scatter = ax.scatter(df['child_coverage'], df['adult_coverage'], 
                     c=df['is_anomaly'], cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal Coverage')
ax.set_xlabel('Child Coverage Ratio (5-17 years)', fontweight='bold', fontsize=12)
ax.set_ylabel('Adult Coverage Ratio (17+ years)', fontweight='bold', fontsize=12)
ax.set_title('Model 1: Child vs Adult Biometric Coverage', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Anomaly Status', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model1_visuals/4_child_adult_coverage.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 4_child_adult_coverage.png")

# Chart 5: Summary Dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Pie chart
ax1 = fig.add_subplot(gs[0, :2])
sizes = [df['is_anomaly'].sum(), (~df['is_anomaly']).sum()]
colors = ['#ff6b6b', '#51cf66']
explode = (0.1, 0)
ax1.pie(sizes, explode=explode, labels=['Anomaly', 'Normal'], colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('District Classification', fontsize=13, fontweight='bold')

# Stats box
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
ANOMALY STATISTICS

Total Districts: {len(df)}
Anomalies: {df['is_anomaly'].sum()}
Normal: {(~df['is_anomaly']).sum()}

Threshold: {threshold:.3f}
Avg Coverage: {df['coverage_ratio'].mean():.3f}
Min Coverage: {df['coverage_ratio'].min():.3f}
Max Coverage: {df['coverage_ratio'].max():.3f}
"""
ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Histogram
ax3 = fig.add_subplot(gs[1, :])
ax3.hist([normal['anomaly_score'], anomaly['anomaly_score']], 
         bins=30, label=['Normal', 'Anomaly'], color=['green', 'red'], alpha=0.6)
ax3.set_xlabel('Anomaly Score', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Top anomalies table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('tight')
ax4.axis('off')
top_10 = df[df['is_anomaly']].nlargest(10, 'anomaly_score')[['district_clean', 'coverage_ratio', 'anomaly_score']]
table_data = [[d, f"{c:.3f}", f"{a:.3f}"] for d, c, a in zip(top_10['district_clean'], top_10['coverage_ratio'], top_10['anomaly_score'])]
table = ax4.table(cellText=table_data, colLabels=['District', 'Coverage', 'Anomaly Score'],
                  cellLoc='left', loc='center', colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(table_data) + 1):
    table[(i, 0)].set_facecolor('#e6f2ff' if i % 2 == 0 else 'white')
    table[(i, 1)].set_facecolor('#e6f2ff' if i % 2 == 0 else 'white')
    table[(i, 2)].set_facecolor('#e6f2ff' if i % 2 == 0 else 'white')
ax4.set_title('Top 10 Anomalous Districts', fontsize=13, fontweight='bold', pad=20)

fig.suptitle('Model 1: Anomaly Detection Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('model1_visuals/5_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 5_summary_dashboard.png")

print("\n" + "="*70)
print("  VISUALIZATION COMPLETE")
print("="*70)
print(f"\n[SUCCESS] Generated 5 charts in 'model1_visuals/' folder")
print(f"[SUCCESS] Detected {df['is_anomaly'].sum()} anomalous districts")
print(f"[SUCCESS] Coverage threshold: {threshold:.3f}")
