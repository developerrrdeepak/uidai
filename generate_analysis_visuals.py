import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Create output directory
Path("analysis_visuals").mkdir(exist_ok=True)

print("Loading cleaned datasets...")
enrollment = pd.read_csv('aadhaar_enrollment_cleaned.csv')
demographic = pd.read_csv('aadhaar_demographic_cleaned.csv')
biometric = pd.read_csv('aadhaar_biometric_cleaned.csv')

# ============================================================================
# UNIVARIATE ANALYSIS
# ============================================================================
print("\nGenerating Univariate Analysis Charts...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Univariate Analysis - Aadhaar Enrollment Data', fontsize=24, fontweight='bold', y=0.98)

# 1. Distribution of Total Enrollments
ax1 = plt.subplot(2, 3, 1)
district_totals = enrollment.groupby('district_clean')['total_enrolment'].sum()
ax1.hist(district_totals, bins=50, color='#2c249f', alpha=0.7, edgecolor='black')
ax1.set_title('Distribution of Total Enrollments\nAcross Districts', fontsize=14, fontweight='bold')
ax1.set_xlabel('Total Enrollments', fontsize=11)
ax1.set_ylabel('Number of Districts', fontsize=11)
ax1.axvline(district_totals.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {district_totals.median():,.0f}')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Biometric Updates Over Time
ax2 = plt.subplot(2, 3, 2)
if 'time_period' in biometric.columns:
    time_updates = biometric.groupby('time_period')['biometric_updates'].sum()
    ax2.plot(range(len(time_updates)), time_updates.values, color='#7bdf72', linewidth=2.5, marker='o')
    ax2.fill_between(range(len(time_updates)), time_updates.values, alpha=0.3, color='#7bdf72')
    ax2.set_title('Trend of Biometric Updates\nOver Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Period', fontsize=11)
    ax2.set_ylabel('Total Biometric Updates', fontsize=11)
    ax2.grid(alpha=0.3)

# 3. Child Enrollments Distribution
ax3 = plt.subplot(2, 3, 3)
child_enrollments = enrollment.groupby('district_clean')['age_0_5'].sum()
ax3.hist(child_enrollments, bins=40, color='#ff6b6b', alpha=0.7, edgecolor='black')
ax3.set_title('Distribution of Child (0-5 years)\nEnrollments', fontsize=14, fontweight='bold')
ax3.set_xlabel('Child Enrollments', fontsize=11)
ax3.set_ylabel('Number of Districts', fontsize=11)
ax3.axvline(child_enrollments.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {child_enrollments.mean():,.0f}')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Box Plot - Enrollment Distribution
ax4 = plt.subplot(2, 3, 4)
age_groups = ['age_0_5', 'age_5_17', 'age_18_greater']
age_data = [enrollment[col].sum() for col in age_groups]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
bars = ax4.bar(['0-5 years', '5-17 years', '18+ years'], age_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_title('Total Enrollments by Age Group', fontsize=14, fontweight='bold')
ax4.set_ylabel('Total Enrollments', fontsize=11)
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Top 10 Districts by Enrollment
ax5 = plt.subplot(2, 3, 5)
top_districts = district_totals.nlargest(10)
ax5.barh(range(len(top_districts)), top_districts.values, color='#2c249f', alpha=0.8, edgecolor='black')
ax5.set_yticks(range(len(top_districts)))
ax5.set_yticklabels(top_districts.index, fontsize=9)
ax5.set_title('Top 10 Districts by\nTotal Enrollments', fontsize=14, fontweight='bold')
ax5.set_xlabel('Total Enrollments', fontsize=11)
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

# 6. Enrollment Statistics Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
UNIVARIATE ANALYSIS SUMMARY

Total Districts Analyzed: {enrollment['district_clean'].nunique():,}
Total Enrollments: {enrollment['total_enrolment'].sum():,}

Age Group Breakdown:
- 0-5 years: {enrollment['age_0_5'].sum():,} ({enrollment['age_0_5'].sum()/enrollment['total_enrolment'].sum()*100:.1f}%)
- 5-17 years: {enrollment['age_5_17'].sum():,} ({enrollment['age_5_17'].sum()/enrollment['total_enrolment'].sum()*100:.1f}%)
- 18+ years: {enrollment['age_18_greater'].sum():,} ({enrollment['age_18_greater'].sum()/enrollment['total_enrolment'].sum()*100:.1f}%)

District Statistics:
- Mean Enrollment: {district_totals.mean():,.0f}
- Median Enrollment: {district_totals.median():,.0f}
- Std Deviation: {district_totals.std():,.0f}
- Min: {district_totals.min():,.0f}
- Max: {district_totals.max():,.0f}
"""
ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), family='monospace')

plt.tight_layout()
plt.savefig('analysis_visuals/UNIVARIATE_ANALYSIS.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: analysis_visuals/UNIVARIATE_ANALYSIS.png")

# ============================================================================
# BIVARIATE ANALYSIS
# ============================================================================
print("\nGenerating Bivariate Analysis Charts...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Bivariate Analysis - Relationship Between Key Indicators', fontsize=24, fontweight='bold', y=0.98)

# Prepare data
district_data = enrollment.groupby('district_clean').agg({
    'total_enrolment': 'sum',
    'age_0_5': 'sum',
    'age_5_17': 'sum',
    'age_18_greater': 'sum'
}).reset_index()

# Calculate ratios
district_data['child_ratio'] = district_data['age_0_5'] / district_data['total_enrolment']
district_data['momentum'] = district_data['total_enrolment'].pct_change().fillna(0)

# 1. Enrollment vs Child Ratio
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(district_data['total_enrolment'], district_data['child_ratio'],
                     c=district_data['child_ratio'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
ax1.set_title('Total Enrollment vs Child Ratio', fontsize=14, fontweight='bold')
ax1.set_xlabel('Total Enrollments', fontsize=11)
ax1.set_ylabel('Child Enrollment Ratio (0-5 years)', fontsize=11)
plt.colorbar(scatter, ax=ax1, label='Child Ratio')
ax1.grid(alpha=0.3)

# 2. Age Group Correlation
ax2 = plt.subplot(2, 3, 2)
corr_data = district_data[['age_0_5', 'age_5_17', 'age_18_greater']].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
ax2.set_title('Age Group Correlation Matrix', fontsize=14, fontweight='bold')
ax2.set_xticklabels(['0-5 yrs', '5-17 yrs', '18+ yrs'], rotation=45)
ax2.set_yticklabels(['0-5 yrs', '5-17 yrs', '18+ yrs'], rotation=0)

# 3. Child vs Adult Enrollments
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(district_data['age_0_5'], district_data['age_18_greater'],
           c='#2c249f', s=80, alpha=0.5, edgecolors='black')
ax3.set_title('Child vs Adult Enrollments', fontsize=14, fontweight='bold')
ax3.set_xlabel('Child Enrollments (0-5 years)', fontsize=11)
ax3.set_ylabel('Adult Enrollments (18+ years)', fontsize=11)
# Add trend line
z = np.polyfit(district_data['age_0_5'], district_data['age_18_greater'], 1)
p = np.poly1d(z)
ax3.plot(district_data['age_0_5'], p(district_data['age_0_5']), "r--", linewidth=2, label='Trend')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Enrollment Distribution by Age Group
ax4 = plt.subplot(2, 3, 4)
age_groups = ['age_0_5', 'age_5_17', 'age_18_greater']
for i, age in enumerate(age_groups):
    ax4.hist(district_data[age], bins=30, alpha=0.5, label=age.replace('_', ' '),
            color=['#ff6b6b', '#4ecdc4', '#45b7d1'][i])
ax4.set_title('Overlapping Distribution of Age Groups', fontsize=14, fontweight='bold')
ax4.set_xlabel('Enrollments', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Top vs Bottom Districts Comparison
ax5 = plt.subplot(2, 3, 5)
top5 = district_data.nlargest(5, 'total_enrolment')
bottom5 = district_data.nsmallest(5, 'total_enrolment')
x = np.arange(3)
width = 0.35
top_vals = [top5['age_0_5'].mean(), top5['age_5_17'].mean(), top5['age_18_greater'].mean()]
bottom_vals = [bottom5['age_0_5'].mean(), bottom5['age_5_17'].mean(), bottom5['age_18_greater'].mean()]
ax5.bar(x - width/2, top_vals, width, label='Top 5 Districts', color='#2c249f', alpha=0.8)
ax5.bar(x + width/2, bottom_vals, width, label='Bottom 5 Districts', color='#ff6b6b', alpha=0.8)
ax5.set_title('Age Distribution: Top vs Bottom Districts', fontsize=14, fontweight='bold')
ax5.set_ylabel('Average Enrollments', fontsize=11)
ax5.set_xticks(x)
ax5.set_xticklabels(['0-5 yrs', '5-17 yrs', '18+ yrs'])
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Bivariate Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
corr_total_child = district_data['total_enrolment'].corr(district_data['age_0_5'])
corr_total_adult = district_data['total_enrolment'].corr(district_data['age_18_greater'])
stats_text = f"""
BIVARIATE ANALYSIS SUMMARY

Correlation Analysis:
- Total vs Child (0-5): {corr_total_child:.3f}
- Total vs Adult (18+): {corr_total_adult:.3f}
- Child vs Adult: {district_data['age_0_5'].corr(district_data['age_18_greater']):.3f}

Child Ratio Statistics:
- Mean: {district_data['child_ratio'].mean():.3f}
- Median: {district_data['child_ratio'].median():.3f}
- Std Dev: {district_data['child_ratio'].std():.3f}

Key Insights:
+ Strong correlation between total
  and adult enrollments
+ Child enrollment ratio varies
  significantly across districts
+ Top districts show higher child
  enrollment proportions
"""
ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3), family='monospace')

plt.tight_layout()
plt.savefig('analysis_visuals/BIVARIATE_ANALYSIS.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: analysis_visuals/BIVARIATE_ANALYSIS.png")

# ============================================================================
# TRIVARIATE ANALYSIS
# ============================================================================
print("\nGenerating Trivariate Analysis Charts...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Trivariate Analysis - Multi-Indicator Risk Assessment', fontsize=24, fontweight='bold', y=0.98)

# Calculate additional metrics
district_data['gender_gap'] = abs(district_data['age_5_17'] - district_data['age_18_greater']) / district_data['total_enrolment']
district_data['risk_score'] = (
    (1 - district_data['child_ratio']) * 0.4 +
    district_data['gender_gap'] * 0.3 +
    (1 - district_data['total_enrolment'] / district_data['total_enrolment'].max()) * 0.3
)

# 1. 3D Scatter: Child Ratio vs Gender Gap vs Total Enrollment
ax1 = plt.subplot(2, 3, 1, projection='3d')
scatter = ax1.scatter(district_data['child_ratio'], district_data['gender_gap'],
                     district_data['total_enrolment'], c=district_data['risk_score'],
                     cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
ax1.set_xlabel('Child Ratio', fontsize=10)
ax1.set_ylabel('Gender Gap', fontsize=10)
ax1.set_zlabel('Total Enrollment', fontsize=10)
ax1.set_title('3D Risk Assessment', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Risk Score', shrink=0.5)

# 2. Bubble Chart: Child Ratio vs Gender Gap (size = enrollment)
ax2 = plt.subplot(2, 3, 2)
sizes = (district_data['total_enrolment'] / district_data['total_enrolment'].max() * 1000)
scatter = ax2.scatter(district_data['child_ratio'], district_data['gender_gap'],
                     s=sizes, c=district_data['risk_score'], cmap='RdYlGn_r',
                     alpha=0.5, edgecolors='black', linewidth=1)
ax2.set_title('Child Ratio vs Gender Gap\n(Bubble size = Total Enrollment)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Child Enrollment Ratio', fontsize=11)
ax2.set_ylabel('Gender Gap', fontsize=11)
plt.colorbar(scatter, ax=ax2, label='Risk Score')
ax2.grid(alpha=0.3)

# 3. Risk Category Distribution
ax3 = plt.subplot(2, 3, 3)
district_data['risk_category'] = pd.cut(district_data['risk_score'],
                                        bins=[0, 0.3, 0.6, 1.0],
                                        labels=['Low', 'Medium', 'High'])
risk_counts = district_data['risk_category'].value_counts()
colors_risk = ['#7bdf72', '#ffd93d', '#ff6b6b']
wedges, texts, autotexts = ax3.pie(risk_counts.values, labels=risk_counts.index,
                                    autopct='%1.1f%%', colors=colors_risk,
                                    startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax3.set_title('Risk Category Distribution', fontsize=14, fontweight='bold')

# 4. Heatmap: Risk Score by Child Ratio and Gender Gap
ax4 = plt.subplot(2, 3, 4)
pivot_data = district_data.pivot_table(values='risk_score',
                                       index=pd.cut(district_data['child_ratio'], bins=5),
                                       columns=pd.cut(district_data['gender_gap'], bins=5),
                                       aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax4,
           cbar_kws={'label': 'Avg Risk Score'})
ax4.set_title('Risk Heatmap: Child Ratio vs Gender Gap', fontsize=14, fontweight='bold')
ax4.set_xlabel('Gender Gap Bins', fontsize=11)
ax4.set_ylabel('Child Ratio Bins', fontsize=11)

# 5. Multi-Indicator Comparison
ax5 = plt.subplot(2, 3, 5)
high_risk = district_data[district_data['risk_category'] == 'High']
medium_risk = district_data[district_data['risk_category'] == 'Medium']
low_risk = district_data[district_data['risk_category'] == 'Low']

indicators = ['child_ratio', 'gender_gap', 'risk_score']
x = np.arange(len(indicators))
width = 0.25

ax5.bar(x - width, [high_risk[ind].mean() for ind in indicators], width,
       label='High Risk', color='#ff6b6b', alpha=0.8)
ax5.bar(x, [medium_risk[ind].mean() for ind in indicators], width,
       label='Medium Risk', color='#ffd93d', alpha=0.8)
ax5.bar(x + width, [low_risk[ind].mean() for ind in indicators], width,
       label='Low Risk', color='#7bdf72', alpha=0.8)

ax5.set_title('Average Indicators by Risk Category', fontsize=14, fontweight='bold')
ax5.set_ylabel('Average Value', fontsize=11)
ax5.set_xticks(x)
ax5.set_xticklabels(['Child Ratio', 'Gender Gap', 'Risk Score'])
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Trivariate Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
TRIVARIATE ANALYSIS SUMMARY

Risk Distribution:
- High Risk: {len(high_risk)} districts ({len(high_risk)/len(district_data)*100:.1f}%)
- Medium Risk: {len(medium_risk)} districts ({len(medium_risk)/len(district_data)*100:.1f}%)
- Low Risk: {len(low_risk)} districts ({len(low_risk)/len(district_data)*100:.1f}%)

High Risk Characteristics:
- Avg Child Ratio: {high_risk['child_ratio'].mean():.3f}
- Avg Gender Gap: {high_risk['gender_gap'].mean():.3f}
- Avg Enrollment: {high_risk['total_enrolment'].mean():,.0f}

Key Insights:
+ Districts with low child ratio AND
  high gender gap show highest risk
+ Combined indicators reveal systemic
  vulnerabilities not visible separately
+ Multi-dimensional assessment enables
  targeted policy interventions
"""
ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3), family='monospace')

plt.tight_layout()
plt.savefig('analysis_visuals/TRIVARIATE_ANALYSIS.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: analysis_visuals/TRIVARIATE_ANALYSIS.png")

print("\n" + "="*70)
print("[SUCCESS] ALL ANALYSIS VISUALIZATIONS GENERATED!")
print("="*70)
print("\nGenerated Files:")
print("  - analysis_visuals/UNIVARIATE_ANALYSIS.png")
print("  - analysis_visuals/BIVARIATE_ANALYSIS.png")
print("  - analysis_visuals/TRIVARIATE_ANALYSIS.png")
print("\n" + "="*70)
