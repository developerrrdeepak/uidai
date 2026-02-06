"""
Model 3: Rule-Based Classification - Single Comprehensive Visualization
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
print("  MODEL 3: RULE-BASED CLASSIFICATION - COMPREHENSIVE DASHBOARD")
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
df['child_coverage'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
df['adult_coverage'] = df['bio_age_17_'] / (df['demo_age_17_'] + 1)
df['risk_score'] = (1 - df['coverage_ratio']).clip(0, 1)

# Rule-based classification
def classify_risk_type(row):
    if row['child_coverage'] < 0.5:
        return 'Child Exclusion Risk'
    elif row['adult_coverage'] < 0.5:
        return 'Update Failure Risk'
    elif row['coverage_ratio'] < 0.3:
        return 'Administrative Disruption'
    elif row['coverage_ratio'] < 0.6:
        return 'Gender Access Barrier'
    else:
        return 'Low Risk'

df['risk_type'] = df.apply(classify_risk_type, axis=1)
df['severity'] = pd.cut(df['risk_score'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])

print(f"[OK] Classified {len(df)} districts")

# Create comprehensive dashboard
print("\n[2/2] Generating comprehensive dashboard...")
os.makedirs('model3_visuals', exist_ok=True)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# Title banner
fig.text(0.5, 0.97, 'MODEL 3: RULE-BASED RISK CLASSIFICATION DASHBOARD', 
         ha='center', fontsize=22, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#9b59b6', edgecolor='none', alpha=0.9),
         color='white')

# 1. Severity KPI Cards
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis('off')
severity_counts = df['severity'].value_counts()
kpis = [
    ('Total Districts', len(df), '#3498db'),
    ('High Severity', severity_counts.get('High', 0), '#e74c3c'),
    ('Medium Severity', severity_counts.get('Medium', 0), '#f39c12'),
    ('Low Severity', severity_counts.get('Low', 0), '#2ecc71')
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

# 2. Risk Type Distribution
ax1 = fig.add_subplot(gs[1, :2])
risk_counts = df['risk_type'].value_counts()
colors_bar = ['#e74c3c', '#f39c12', '#e67e22', '#3498db', '#2ecc71']
bars = ax1.bar(range(len(risk_counts)), risk_counts.values, 
              color=colors_bar[:len(risk_counts)], alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_xticks(range(len(risk_counts)))
ax1.set_xticklabels([t[:20] for t in risk_counts.index], rotation=45, ha='right', fontsize=10)
ax1.set_ylabel('Number of Districts', fontweight='bold', fontsize=12)
ax1.set_title('Risk Type Distribution', fontsize=14, fontweight='bold', pad=10)
for i, v in enumerate(risk_counts.values):
    ax1.text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=11)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.set_facecolor('#f8f9fa')

# 3. Severity by Risk Type (Stacked)
ax2 = fig.add_subplot(gs[1, 2:])
severity_by_type = pd.crosstab(df['risk_type'], df['severity'])
severity_by_type.plot(kind='bar', stacked=True, ax=ax2, 
                      color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Risk Type', fontweight='bold', fontsize=12)
ax2.set_ylabel('Count', fontweight='bold', fontsize=12)
ax2.set_title('Severity Distribution by Risk Type', fontsize=14, fontweight='bold', pad=10)
ax2.legend(title='Severity', fontsize=10, framealpha=0.9)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_facecolor('#f8f9fa')

# 4. Top 10 High-Risk Districts
ax3 = fig.add_subplot(gs[2, :2])
top_10 = df.nlargest(10, 'risk_score')
risk_type_colors = {
    'Child Exclusion Risk': '#e74c3c',
    'Update Failure Risk': '#f39c12',
    'Administrative Disruption': '#e67e22',
    'Gender Access Barrier': '#3498db',
    'Low Risk': '#2ecc71'
}
colors_map = [risk_type_colors.get(str(t), '#95a5a6') for t in top_10['risk_type']]
bars = ax3.barh(range(len(top_10)), top_10['risk_score'], color=colors_map, 
                edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(top_10)))
ax3.set_yticklabels([d[:30] for d in top_10['district_clean']], fontsize=10)
ax3.set_xlabel('Risk Score', fontweight='bold', fontsize=12)
ax3.set_title('Top 10 High-Risk Districts (Color = Risk Type)', fontsize=14, fontweight='bold', pad=10)
ax3.invert_yaxis()
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax3.text(row['risk_score'] + 0.01, i, f"{row['risk_score']:.3f}", 
            va='center', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
ax3.set_facecolor('#f8f9fa')

# 5. Classification Statistics & Legend
ax4 = fig.add_subplot(gs[2, 2:])
ax4.axis('off')

# Stats box
stats_text = f"""
CLASSIFICATION STATISTICS

Total Districts: {len(df)}

SEVERITY LEVELS:
  High: {severity_counts.get('High', 0)} ({severity_counts.get('High', 0)/len(df)*100:.1f}%)
  Medium: {severity_counts.get('Medium', 0)} ({severity_counts.get('Medium', 0)/len(df)*100:.1f}%)
  Low: {severity_counts.get('Low', 0)} ({severity_counts.get('Low', 0)/len(df)*100:.1f}%)

RISK TYPES:
"""
for risk_type, count in risk_counts.head(5).items():
    stats_text += f"  {risk_type[:25]}: {count}\n"

stats_text += f"""
Average Risk Score: {df['risk_score'].mean():.3f}
Max Risk Score: {df['risk_score'].max():.3f}

RULE-BASED LOGIC:
  Child Coverage < 0.5 → Child Exclusion
  Adult Coverage < 0.5 → Update Failure
  Coverage < 0.3 → Admin Disruption
  Coverage < 0.6 → Gender Barrier
  Otherwise → Low Risk
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8daef', edgecolor='#9b59b6', linewidth=2))

# Footer
fig.text(0.5, 0.01, 'Data Source: aadhaar_biometric_cleaned.csv & aadhaar_demographic_cleaned.csv | Rule-Based Classification System',
         ha='center', fontsize=9, style='italic', color='#6c757d')

plt.savefig('model3_visuals/MODEL3_COMPREHENSIVE_DASHBOARD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("[SUCCESS] Generated: MODEL3_COMPREHENSIVE_DASHBOARD.png")
print(f"[SUCCESS] Classified {len(df)} districts into {len(risk_counts)} risk types")
