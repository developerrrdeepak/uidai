"""
Model 2: Risk Scoring Visualization Generator
Generates comprehensive charts for risk scoring using cleaned data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("  MODEL 2: RISK SCORING VISUALIZATIONS")
print("="*70)

# Load cleaned data
print("\n[1/3] Loading cleaned CSV files...")
bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')
print(f"[OK] Biometric: {len(bio_df)} rows, {bio_df['district_clean'].nunique()} districts")
print(f"[OK] Demographic: {len(demo_df)} rows, {demo_df['district_clean'].nunique()} districts")

# Aggregate by district
print("\n[2/3] Calculating risk scores...")
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

# Calculate risk components
df['coverage_ratio'] = df['total_bio'] / (df['total_demo'] + 1)
df['child_ratio'] = df['bio_age_5_17'] / (df['total_bio'] + 1)
df['adult_ratio'] = df['bio_age_17_'] / (df['total_bio'] + 1)

# Risk Score = 1 - Coverage Ratio (higher score = higher risk)
df['risk_score'] = 1 - df['coverage_ratio']
df['risk_score'] = df['risk_score'].clip(0, 1)

# Risk Classification
df['risk_level'] = pd.cut(df['risk_score'], 
                          bins=[0, 0.35, 0.6, 1.0],
                          labels=['Low', 'Medium', 'High'])

print(f"[OK] Processed {len(df)} districts")
print(f"[OK] High Risk: {(df['risk_level']=='High').sum()}")
print(f"[OK] Medium Risk: {(df['risk_level']=='Medium').sum()}")
print(f"[OK] Low Risk: {(df['risk_level']=='Low').sum()}")

# Create output directory
os.makedirs('model2_visuals', exist_ok=True)

print("\n[3/3] Generating visualizations...")

# Chart 1: Risk Score Distribution
fig, ax = plt.subplots(figsize=(12, 6))
colors = {'Low': '#51cf66', 'Medium': '#ffd43b', 'High': '#ff6b6b'}
for level in ['Low', 'Medium', 'High']:
    data = df[df['risk_level'] == level]['risk_score']
    ax.hist(data, bins=30, alpha=0.6, label=level, color=colors[level])
ax.set_xlabel('Risk Score', fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Districts', fontweight='bold', fontsize=12)
ax.set_title('Model 2: Risk Score Distribution by Risk Level', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model2_visuals/1_risk_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 1_risk_distribution.png")

# Chart 2: Top 20 High-Risk Districts
high_risk = df.nlargest(20, 'risk_score')
fig, ax = plt.subplots(figsize=(14, 8))
colors_map = high_risk['risk_level'].map({'High': '#ff6b6b', 'Medium': '#ffd43b', 'Low': '#51cf66'})
bars = ax.barh(range(len(high_risk)), high_risk['risk_score'], color=colors_map, alpha=0.8)
ax.set_yticks(range(len(high_risk)))
ax.set_yticklabels(high_risk['district_clean'], fontsize=10)
ax.set_xlabel('Risk Score', fontweight='bold', fontsize=12)
ax.set_title('Model 2: Top 20 High-Risk Districts', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (idx, row) in enumerate(high_risk.iterrows()):
    ax.text(row['risk_score'] + 0.01, i, f"{row['risk_score']:.3f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig('model2_visuals/2_top_high_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 2_top_high_risk.png")

# Chart 3: Risk Level Pie Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
risk_counts = df['risk_level'].value_counts()
colors_pie = ['#51cf66', '#ffd43b', '#ff6b6b']
explode = (0.05, 0.05, 0.1)
ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
        colors=colors_pie, explode=explode, shadow=True, startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Risk Level Distribution', fontsize=13, fontweight='bold')

# Box plot by risk level
data_box = [df[df['risk_level']=='Low']['risk_score'],
            df[df['risk_level']=='Medium']['risk_score'],
            df[df['risk_level']=='High']['risk_score']]
bp = ax2.boxplot(data_box, tick_labels=['Low', 'Medium', 'High'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#51cf66', '#ffd43b', '#ff6b6b']):
    patch.set_facecolor(color)
ax2.set_ylabel('Risk Score', fontweight='bold', fontsize=12)
ax2.set_title('Risk Score by Level', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model2_visuals/3_risk_level_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 3_risk_level_analysis.png")

# Chart 4: Coverage vs Risk Scatter
fig, ax = plt.subplots(figsize=(12, 8))
df_plot = df.dropna(subset=['risk_level'])
colors_scatter = df_plot['risk_level'].map({'Low': '#51cf66', 'Medium': '#ffd43b', 'High': '#ff6b6b'})
scatter = ax.scatter(df_plot['total_demo'], df_plot['total_bio'], 
                     c=colors_scatter, s=50, alpha=0.6, edgecolors='black')
ax.plot([0, df['total_demo'].max()], [0, df['total_demo'].max()], 
        'k--', alpha=0.3, linewidth=2, label='Perfect Coverage')
ax.set_xlabel('Demographic Coverage', fontweight='bold', fontsize=12)
ax.set_ylabel('Biometric Coverage', fontweight='bold', fontsize=12)
ax.set_title('Model 2: Biometric vs Demographic Coverage (Color = Risk Level)', fontsize=14, fontweight='bold')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
           for l, c in [('Low Risk', '#51cf66'), ('Medium Risk', '#ffd43b'), ('High Risk', '#ff6b6b')]]
ax.legend(handles=handles, fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model2_visuals/4_coverage_vs_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 4_coverage_vs_risk.png")

# Chart 5: Risk Components Heatmap
top_30 = df.nlargest(30, 'risk_score')[['district_clean', 'risk_score', 'coverage_ratio', 'child_ratio', 'adult_ratio']]
fig, ax = plt.subplots(figsize=(10, 12))
data_heatmap = top_30[['risk_score', 'coverage_ratio', 'child_ratio', 'adult_ratio']].values
im = ax.imshow(data_heatmap, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels(['Risk Score', 'Coverage', 'Child Ratio', 'Adult Ratio'], rotation=45, ha='right')
ax.set_yticks(range(len(top_30)))
ax.set_yticklabels(top_30['district_clean'], fontsize=8)
ax.set_title('Model 2: Risk Components Heatmap (Top 30 Districts)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score', fontweight='bold')
for i in range(len(top_30)):
    for j in range(4):
        text = ax.text(j, i, f'{data_heatmap[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=7)
plt.tight_layout()
plt.savefig('model2_visuals/5_risk_components_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 5_risk_components_heatmap.png")

# Chart 6: Comprehensive Dashboard
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Risk distribution histogram
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist([df[df['risk_level']=='Low']['risk_score'],
          df[df['risk_level']=='Medium']['risk_score'],
          df[df['risk_level']=='High']['risk_score']], 
         bins=30, label=['Low', 'Medium', 'High'], 
         color=['#51cf66', '#ffd43b', '#ff6b6b'], alpha=0.6, stacked=True)
ax1.set_xlabel('Risk Score', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Risk Score Distribution', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Stats box
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
RISK SCORING STATISTICS

Total Districts: {len(df)}
High Risk: {(df['risk_level']=='High').sum()}
Medium Risk: {(df['risk_level']=='Medium').sum()}
Low Risk: {(df['risk_level']=='Low').sum()}

Avg Risk Score: {df['risk_score'].mean():.3f}
Max Risk Score: {df['risk_score'].max():.3f}
Min Risk Score: {df['risk_score'].min():.3f}

Avg Coverage: {df['coverage_ratio'].mean():.3f}
"""
ax2.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Risk level bar chart
ax3 = fig.add_subplot(gs[1, 0])
risk_counts.plot(kind='bar', ax=ax3, color=['#51cf66', '#ffd43b', '#ff6b6b'], alpha=0.8)
ax3.set_xlabel('Risk Level', fontweight='bold')
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_title('Districts by Risk Level', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=0)
for i, v in enumerate(risk_counts):
    ax3.text(i, v + 5, str(v), ha='center', fontweight='bold')

# Coverage distribution
ax4 = fig.add_subplot(gs[1, 1:])
ax4.hist(df['coverage_ratio'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(df['coverage_ratio'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['coverage_ratio'].mean():.3f}")
ax4.set_xlabel('Coverage Ratio', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.set_title('Coverage Ratio Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Top 10 high-risk table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('tight')
ax5.axis('off')
top_10 = df.nlargest(10, 'risk_score')[['district_clean', 'risk_score', 'risk_level', 'coverage_ratio']]
table_data = [[d, f"{r:.3f}", l, f"{c:.3f}"] 
              for d, r, l, c in zip(top_10['district_clean'], top_10['risk_score'], 
                                     top_10['risk_level'], top_10['coverage_ratio'])]
table = ax5.table(cellText=table_data, 
                  colLabels=['District', 'Risk Score', 'Risk Level', 'Coverage'],
                  cellLoc='left', loc='center', colWidths=[0.4, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(table_data) + 1):
    for j in range(4):
        cell = table[(i, j)]
        cell.set_facecolor('#ffe6e6' if i > 0 and table_data[i-1][2] == 'High' else 
                          ('#fff4e6' if i > 0 and table_data[i-1][2] == 'Medium' else 
                           ('#e6ffe6' if i > 0 else 'lightgray')))
ax5.set_title('Top 10 High-Risk Districts', fontsize=13, fontweight='bold', pad=20)

fig.suptitle('Model 2: Risk Scoring Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('model2_visuals/6_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated: 6_comprehensive_dashboard.png")

print("\n" + "="*70)
print("  VISUALIZATION COMPLETE")
print("="*70)
print(f"\n[SUCCESS] Generated 6 charts in 'model2_visuals/' folder")
print(f"[SUCCESS] High Risk: {(df['risk_level']=='High').sum()} districts")
print(f"[SUCCESS] Medium Risk: {(df['risk_level']=='Medium').sum()} districts")
print(f"[SUCCESS] Low Risk: {(df['risk_level']=='Low').sum()} districts")
print(f"[SUCCESS] Average Risk Score: {df['risk_score'].mean():.3f}")
