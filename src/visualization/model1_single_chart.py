"""
Model 1: Anomaly Detection - Single Comprehensive Visualization
Creative dashboard showing all key insights in one chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("  MODEL 1: ANOMALY DETECTION - COMPREHENSIVE DASHBOARD")
print("="*70)

# Load data
print("\n[1/2] Loading cleaned CSV files...")
bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')

# Process
bio_agg = bio_df.groupby('district_clean').agg({'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'}).reset_index()
bio_agg['total_bio'] = bio_agg['bio_age_5_17'] + bio_agg['bio_age_17_']

demo_agg = demo_df.groupby('district_clean').agg({'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'}).reset_index()
demo_agg['total_demo'] = demo_agg['demo_age_5_17'] + demo_agg['demo_age_17_']

df = pd.merge(bio_agg, demo_agg, on='district_clean', how='outer').fillna(0)
df['coverage_ratio'] = df['total_bio'] / (df['total_demo'] + 1)
df['anomaly_score'] = 1 - df['coverage_ratio']
threshold = df['coverage_ratio'].quantile(0.25)
df['is_anomaly'] = df['coverage_ratio'] < threshold

print(f"[OK] Processed {len(df)} districts, detected {df['is_anomaly'].sum()} anomalies")

# Create comprehensive dashboard
print("\n[2/2] Generating comprehensive dashboard...")
os.makedirs('model1_visuals', exist_ok=True)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# Title banner
fig.text(0.5, 0.97, 'MODEL 1: ANOMALY DETECTION DASHBOARD', 
         ha='center', fontsize=22, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#667eea', edgecolor='none', alpha=0.9),
         color='white')

# 1. Main KPI Cards (Top)
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis('off')
kpis = [
    ('Total Districts', len(df), '#4dabf7'),
    ('Anomalies Detected', df['is_anomaly'].sum(), '#ff6b6b'),
    ('Normal Districts', (~df['is_anomaly']).sum(), '#51cf66'),
    ('Anomaly Rate', f"{(df['is_anomaly'].sum()/len(df)*100):.1f}%", '#ffd43b')
]
for i, (label, value, color) in enumerate(kpis):
    x = 0.125 + i * 0.22
    rect = Rectangle((x-0.08, 0.3), 0.16, 0.5, transform=ax_kpi.transAxes,
                     facecolor=color, alpha=0.2, edgecolor=color, linewidth=3)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x, 0.7, str(value), ha='center', va='center', fontsize=28, fontweight='bold',
               transform=ax_kpi.transAxes, color=color)
    ax_kpi.text(x, 0.4, label, ha='center', va='center', fontsize=11, fontweight='bold',
               transform=ax_kpi.transAxes, color='#2d3436')

# 2. Coverage Distribution with Threshold
ax1 = fig.add_subplot(gs[1, :2])
normal = df[~df['is_anomaly']]['coverage_ratio']
anomaly = df[df['is_anomaly']]['coverage_ratio']
ax1.hist([normal, anomaly], bins=40, label=['Normal', 'Anomaly'], 
         color=['#51cf66', '#ff6b6b'], alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.axvline(threshold, color='#ffd43b', linestyle='--', linewidth=3, label=f'Threshold: {threshold:.3f}')
ax1.set_xlabel('Biometric Coverage Ratio', fontweight='bold', fontsize=12)
ax1.set_ylabel('Number of Districts', fontweight='bold', fontsize=12)
ax1.set_title('Coverage Distribution & Anomaly Threshold', fontsize=14, fontweight='bold', pad=10)
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_facecolor('#f8f9fa')

# 3. Scatter Plot: Bio vs Demo Coverage
ax2 = fig.add_subplot(gs[1, 2:])
colors = ['#ff6b6b' if a else '#51cf66' for a in df['is_anomaly']]
ax2.scatter(df['total_demo'], df['total_bio'], c=colors, s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
max_val = max(df['total_demo'].max(), df['total_bio'].max())
ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=2, label='Perfect Coverage')
ax2.set_xlabel('Demographic Coverage', fontweight='bold', fontsize=12)
ax2.set_ylabel('Biometric Coverage', fontweight='bold', fontsize=12)
ax2.set_title('Biometric vs Demographic Coverage', fontsize=14, fontweight='bold', pad=10)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
           for l, c in [('Normal', '#51cf66'), ('Anomaly', '#ff6b6b')]]
ax2.legend(handles=handles, fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('#f8f9fa')

# 4. Top 10 Anomalies Bar Chart
ax3 = fig.add_subplot(gs[2, :2])
top_10 = df[df['is_anomaly']].nlargest(10, 'anomaly_score')
colors_gradient = plt.cm.Reds(np.linspace(0.5, 0.9, len(top_10)))
bars = ax3.barh(range(len(top_10)), top_10['anomaly_score'], color=colors_gradient, 
                edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(top_10)))
ax3.set_yticklabels(top_10['district_clean'], fontsize=10)
ax3.set_xlabel('Anomaly Score', fontweight='bold', fontsize=12)
ax3.set_title('Top 10 Anomalous Districts', fontsize=14, fontweight='bold', pad=10)
ax3.invert_yaxis()
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax3.text(row['anomaly_score'] + 0.02, i, f"{row['anomaly_score']:.3f}", 
            va='center', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
ax3.set_facecolor('#f8f9fa')

# 5. Anomaly Statistics Box
ax4 = fig.add_subplot(gs[2, 2:])
ax4.axis('off')
stats_text = f"""
ANOMALY DETECTION STATISTICS

Total Districts Analyzed: {len(df)}
Anomalies Detected: {df['is_anomaly'].sum()}
Normal Districts: {(~df['is_anomaly']).sum()}
Detection Rate: {(df['is_anomaly'].sum()/len(df)*100):.1f}%

Coverage Threshold: {threshold:.3f}
Average Coverage (All): {df['coverage_ratio'].mean():.3f}
Average Coverage (Normal): {normal.mean():.3f}
Average Coverage (Anomaly): {anomaly.mean():.3f}

Highest Anomaly Score: {df['anomaly_score'].max():.3f}
Lowest Anomaly Score: {df['anomaly_score'].min():.3f}

Child Coverage (5-17): {(df['bio_age_5_17'].sum() / (df['demo_age_5_17'].sum() + 1)):.3f}
Adult Coverage (17+): {(df['bio_age_17_'].sum() / (df['demo_age_17_'].sum() + 1)):.3f}
"""
ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2))

# Footer
fig.text(0.5, 0.01, 'Data Source: aadhaar_biometric_cleaned.csv & aadhaar_demographic_cleaned.csv | Anomaly Detection using Coverage Ratio Analysis',
         ha='center', fontsize=9, style='italic', color='#6c757d')

plt.savefig('model1_visuals/MODEL1_COMPREHENSIVE_DASHBOARD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("[SUCCESS] Generated: MODEL1_COMPREHENSIVE_DASHBOARD.png")
print(f"[SUCCESS] Detected {df['is_anomaly'].sum()} anomalies from {len(df)} districts")
