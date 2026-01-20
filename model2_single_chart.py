"""
Model 2: Risk Scoring - Single Comprehensive Visualization
Creative dashboard showing all key insights in one chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Wedge
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("  MODEL 2: RISK SCORING - COMPREHENSIVE DASHBOARD")
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
df['risk_score'] = (1 - df['coverage_ratio']).clip(0, 1)
df['risk_level'] = pd.cut(df['risk_score'], bins=[0, 0.35, 0.6, 1.0], labels=['Low', 'Medium', 'High'])

print(f"[OK] Scored {len(df)} districts")

# Create comprehensive dashboard
print("\n[2/2] Generating comprehensive dashboard...")
os.makedirs('model2_visuals', exist_ok=True)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# Title banner
fig.text(0.5, 0.97, 'MODEL 2: RISK SCORING DASHBOARD', 
         ha='center', fontsize=22, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#e74c3c', edgecolor='none', alpha=0.9),
         color='white')

# 1. Risk Level KPI Cards
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis('off')
risk_counts = df['risk_level'].value_counts()
kpis = [
    ('Total Districts', len(df), '#95a5a6'),
    ('High Risk', risk_counts.get('High', 0), '#e74c3c'),
    ('Medium Risk', risk_counts.get('Medium', 0), '#f39c12'),
    ('Low Risk', risk_counts.get('Low', 0), '#27ae60')
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

# 2. Risk Score Distribution
ax1 = fig.add_subplot(gs[1, :2])
for level, color in [('Low', '#27ae60'), ('Medium', '#f39c12'), ('High', '#e74c3c')]:
    data = df[df['risk_level'] == level]['risk_score']
    if len(data) > 0:
        ax1.hist(data, bins=25, alpha=0.7, label=level, color=color, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Risk Score', fontweight='bold', fontsize=12)
ax1.set_ylabel('Number of Districts', fontweight='bold', fontsize=12)
ax1.set_title('Risk Score Distribution by Level', fontsize=14, fontweight='bold', pad=10)
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_facecolor('#f8f9fa')

# 3. Risk Level Donut Chart
ax2 = fig.add_subplot(gs[1, 2:])
colors_pie = ['#27ae60', '#f39c12', '#e74c3c']
sizes = [risk_counts.get(l, 0) for l in ['Low', 'Medium', 'High']]
wedges, texts, autotexts = ax2.pie(sizes, labels=['Low', 'Medium', 'High'], autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                                    textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
ax2.set_title('Risk Level Distribution', fontsize=14, fontweight='bold', pad=10)
centre_circle = plt.Circle((0, 0), 0.60, fc='white', linewidth=2, edgecolor='#2d3436')
ax2.add_artist(centre_circle)
ax2.text(0, 0, f'{len(df)}\nDistricts', ha='center', va='center', fontsize=16, fontweight='bold', color='#2d3436')

# 4. Top 12 High-Risk Districts
ax3 = fig.add_subplot(gs[2, :3])
top_12 = df.nlargest(12, 'risk_score')
colors_gradient = [{'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}.get(str(l), '#95a5a6') 
                   for l in top_12['risk_level']]
bars = ax3.barh(range(len(top_12)), top_12['risk_score'], color=colors_gradient, 
                edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(top_12)))
ax3.set_yticklabels([d[:30] for d in top_12['district_clean']], fontsize=9)
ax3.set_xlabel('Risk Score', fontweight='bold', fontsize=12)
ax3.set_title('Top 12 High-Risk Districts', fontsize=14, fontweight='bold', pad=10)
ax3.invert_yaxis()
for i, (idx, row) in enumerate(top_12.iterrows()):
    ax3.text(row['risk_score'] + 0.01, i, f"{row['risk_score']:.3f}", 
            va='center', fontsize=8, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
ax3.set_facecolor('#f8f9fa')

# 5. Risk Scoring Statistics
ax4 = fig.add_subplot(gs[2, 3])
ax4.axis('off')
stats_text = f"""
RISK SCORING STATS

Districts: {len(df)}

HIGH RISK: {risk_counts.get('High', 0)}
  ({risk_counts.get('High', 0)/len(df)*100:.1f}%)

MEDIUM RISK: {risk_counts.get('Medium', 0)}
  ({risk_counts.get('Medium', 0)/len(df)*100:.1f}%)

LOW RISK: {risk_counts.get('Low', 0)}
  ({risk_counts.get('Low', 0)/len(df)*100:.1f}%)

Avg Risk Score:
  {df['risk_score'].mean():.3f}

Max Risk Score:
  {df['risk_score'].max():.3f}

Min Risk Score:
  {df['risk_score'].min():.3f}

Std Deviation:
  {df['risk_score'].std():.3f}
"""
ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffe6e6', edgecolor='#e74c3c', linewidth=2))

# Footer
fig.text(0.5, 0.01, 'Data Source: aadhaar_biometric_cleaned.csv & aadhaar_demographic_cleaned.csv | Risk Scoring: 1 - Coverage Ratio',
         ha='center', fontsize=9, style='italic', color='#6c757d')

plt.savefig('model2_visuals/MODEL2_COMPREHENSIVE_DASHBOARD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("[SUCCESS] Generated: MODEL2_COMPREHENSIVE_DASHBOARD.png")
print(f"[SUCCESS] Scored {len(df)} districts: High={risk_counts.get('High', 0)}, Medium={risk_counts.get('Medium', 0)}, Low={risk_counts.get('Low', 0)}")
