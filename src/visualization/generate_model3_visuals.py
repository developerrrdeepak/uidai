"""
Model 3: Rule-Based Risk Classification Visualization Generator
Generates comprehensive charts for risk classification using cleaned data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("  MODEL 3: RULE-BASED RISK CLASSIFICATION VISUALIZATIONS")
print("="*70)

# Load cleaned data
print("\n[1/3] Loading cleaned CSV files...")
bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')
print(f"[OK] Biometric: {len(bio_df)} rows, {bio_df['district_clean'].nunique()} districts")
print(f"[OK] Demographic: {len(demo_df)} rows, {demo_df['district_clean'].nunique()} districts")

# Aggregate by district
print("\n[2/3] Applying rule-based classification...")
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

# Calculate metrics
df['coverage_ratio'] = df['total_bio'] / (df['total_demo'] + 1)
df['child_coverage'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
df['adult_coverage'] = df['bio_age_17_'] / (df['demo_age_17_'] + 1)
df['risk_score'] = 1 - df['coverage_ratio']
df['risk_score'] = df['risk_score'].clip(0, 1)

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

# Severity classification
df['severity'] = pd.cut(df['risk_score'], 
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['Low', 'Medium', 'High'])

print(f"[OK] Classified {len(df)} districts")
print(f"[OK] High Severity: {(df['severity']=='High').sum()}")
print(f"[OK] Medium Severity: {(df['severity']=='Medium').sum()}")
print(f"[OK] Low Severity: {(df['severity']=='Low').sum()}")

# Create output directory
os.makedirs('model3_visuals', exist_ok=True)

print("\n[3/3] Generating visualizations...")

# Chart 1: Risk Type Distribution
fig, ax = plt.subplots(figsize=(12, 7))
risk_counts = df['risk_type'].value_counts()
colors = ['#ff6b6b', '#ffd43b', '#51cf66', '#4dabf7', '#845ef7']
bars = ax.bar(range(len(risk_counts)), risk_counts.values, color=colors[:len(risk_counts)], alpha=0.8)
ax.set_xticks(range(len(risk_counts)))
ax.set_xticklabels(risk_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Districts', fontweight='bold', fontsize=12)
ax.set_title('Model 3: Risk Type Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(risk_counts.values):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model3_visuals/1_risk_type_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 1_risk_type_distribution.png")

# Chart 2: Severity Distribution by Risk Type
fig, ax = plt.subplots(figsize=(14, 7))
severity_by_type = pd.crosstab(df['risk_type'], df['severity'])
severity_by_type.plot(kind='bar', stacked=True, ax=ax, 
                      color=['#51cf66', '#ffd43b', '#ff6b6b'], alpha=0.8)
ax.set_xlabel('Risk Type', fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Districts', fontweight='bold', fontsize=12)
ax.set_title('Model 3: Severity Distribution by Risk Type', fontsize=14, fontweight='bold')
ax.legend(title='Severity', fontsize=11)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('model3_visuals/2_severity_by_risk_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 2_severity_by_risk_type.png")

# Chart 3: Risk Type Pie Chart with Severity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
risk_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', 
                colors=colors[:len(risk_counts)], startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_ylabel('')
ax1.set_title('Risk Type Distribution', fontsize=13, fontweight='bold')

severity_counts = df['severity'].value_counts()
severity_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%',
                    colors=['#51cf66', '#ffd43b', '#ff6b6b'], startangle=90,
                    textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_ylabel('')
ax2.set_title('Severity Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('model3_visuals/3_pie_charts.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 3_pie_charts.png")

# Chart 4: Top 20 Districts by Risk Type
top_20 = df.nlargest(20, 'risk_score')
fig, ax = plt.subplots(figsize=(14, 9))
risk_type_colors = {
    'Child Exclusion Risk': '#ff6b6b',
    'Update Failure Risk': '#ffd43b',
    'Administrative Disruption': '#ff8787',
    'Gender Access Barrier': '#ffa94d',
    'Low Risk': '#51cf66'
}
colors_map = top_20['risk_type'].map(risk_type_colors)
bars = ax.barh(range(len(top_20)), top_20['risk_score'], color=colors_map, alpha=0.8)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['district_clean'], fontsize=10)
ax.set_xlabel('Risk Score', fontweight='bold', fontsize=12)
ax.set_title('Model 3: Top 20 Districts by Risk Score (Color = Risk Type)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (idx, row) in enumerate(top_20.iterrows()):
    ax.text(row['risk_score'] + 0.01, i, f"{row['risk_score']:.3f}", va='center', fontsize=9)
handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.8) for c in risk_type_colors.values()]
ax.legend(handles, risk_type_colors.keys(), loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('model3_visuals/4_top_districts_by_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 4_top_districts_by_type.png")

# Chart 5: Coverage Analysis by Risk Type
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
risk_types = df['risk_type'].unique()[:4]

for idx, risk_type in enumerate(risk_types):
    ax = axes[idx // 2, idx % 2]
    data = df[df['risk_type'] == risk_type].dropna(subset=['severity'])
    colors_scatter = data['severity'].map({'Low': '#51cf66', 'Medium': '#ffd43b', 'High': '#ff6b6b'})
    ax.scatter(data['child_coverage'], data['adult_coverage'], 
              c=colors_scatter, s=50, alpha=0.6, edgecolors='black')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Child Coverage', fontweight='bold')
    ax.set_ylabel('Adult Coverage', fontweight='bold')
    ax.set_title(f'{risk_type}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, max(1.1, data['child_coverage'].max() + 0.1))
    ax.set_ylim(-0.1, max(1.1, data['adult_coverage'].max() + 0.1))

plt.suptitle('Model 3: Coverage Analysis by Risk Type', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('model3_visuals/5_coverage_by_risk_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 5_coverage_by_risk_type.png")

# Chart 6: Comprehensive Dashboard
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# Risk type bar chart
ax1 = fig.add_subplot(gs[0, :2])
risk_counts.plot(kind='bar', ax=ax1, color=colors[:len(risk_counts)], alpha=0.8)
ax1.set_xlabel('Risk Type', fontweight='bold')
ax1.set_ylabel('Count', fontweight='bold')
ax1.set_title('Risk Type Distribution', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(risk_counts.values):
    ax1.text(i, v + 3, str(v), ha='center', fontweight='bold')

# Stats box
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
CLASSIFICATION STATISTICS

Total Districts: {len(df)}

SEVERITY:
High: {(df['severity']=='High').sum()}
Medium: {(df['severity']=='Medium').sum()}
Low: {(df['severity']=='Low').sum()}

RISK TYPES:
{chr(10).join([f'{k}: {v}' for k, v in risk_counts.head(5).items()])}

Avg Risk Score: {df['risk_score'].mean():.3f}
"""
ax2.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Severity stacked bar
ax3 = fig.add_subplot(gs[1, :])
severity_by_type.plot(kind='bar', stacked=True, ax=ax3,
                      color=['#51cf66', '#ffd43b', '#ff6b6b'], alpha=0.8)
ax3.set_xlabel('Risk Type', fontweight='bold')
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_title('Severity by Risk Type', fontsize=13, fontweight='bold')
ax3.legend(title='Severity')
ax3.tick_params(axis='x', rotation=45)

# Heatmap
ax4 = fig.add_subplot(gs[2, :])
top_15 = df.nlargest(15, 'risk_score')[['district_clean', 'risk_score', 'coverage_ratio', 'child_coverage', 'adult_coverage']]
data_heatmap = top_15[['risk_score', 'coverage_ratio', 'child_coverage', 'adult_coverage']].values
im = ax4.imshow(data_heatmap, cmap='RdYlGn_r', aspect='auto')
ax4.set_xticks(range(4))
ax4.set_xticklabels(['Risk Score', 'Coverage', 'Child Cov', 'Adult Cov'], rotation=45, ha='right')
ax4.set_yticks(range(len(top_15)))
ax4.set_yticklabels(top_15['district_clean'], fontsize=8)
ax4.set_title('Risk Metrics Heatmap (Top 15)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('Score', fontweight='bold')

# Top 10 table
ax5 = fig.add_subplot(gs[3, :])
ax5.axis('tight')
ax5.axis('off')
top_10 = df.nlargest(10, 'risk_score')[['district_clean', 'risk_type', 'severity', 'risk_score']]
table_data = [[d[:25], t[:20], s, f"{r:.3f}"] 
              for d, t, s, r in zip(top_10['district_clean'], top_10['risk_type'], 
                                     top_10['severity'], top_10['risk_score'])]
table = ax5.table(cellText=table_data, 
                  colLabels=['District', 'Risk Type', 'Severity', 'Risk Score'],
                  cellLoc='left', loc='center', colWidths=[0.3, 0.3, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(table_data) + 1):
    for j in range(4):
        cell = table[(i, j)]
        if i > 0:
            severity = table_data[i-1][2]
            cell.set_facecolor('#ffe6e6' if severity == 'High' else 
                              ('#fff4e6' if severity == 'Medium' else '#e6ffe6'))
        else:
            cell.set_facecolor('lightgray')
ax5.set_title('Top 10 High-Risk Districts', fontsize=13, fontweight='bold', pad=20)

fig.suptitle('Model 3: Rule-Based Risk Classification Dashboard', fontsize=16, fontweight='bold', y=0.99)
plt.savefig('model3_visuals/6_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 6_comprehensive_dashboard.png")

print("\n" + "="*70)
print("  VISUALIZATION COMPLETE")
print("="*70)
print(f"\n[SUCCESS] Generated 6 charts in 'model3_visuals/' folder")
print(f"[SUCCESS] Classified {len(df)} districts into {len(risk_counts)} risk types")
print(f"[SUCCESS] High Severity: {(df['severity']=='High').sum()} districts")
print(f"[SUCCESS] Medium Severity: {(df['severity']=='Medium').sum()} districts")
print(f"[SUCCESS] Low Severity: {(df['severity']=='Low').sum()} districts")
