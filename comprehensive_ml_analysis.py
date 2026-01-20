import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('ml_analysis_results', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("  COMPREHENSIVE ML ANALYSIS - UIDAI RISK PREDICTION")
print("="*70)

# Load data
print("\n[1/8] Loading data...")
try:
    bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
    demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')
    print(f"+ Biometric: {len(bio_df)} rows")
    print(f"+ Demographic: {len(demo_df)} rows")
    
    enrol_files = ['api_data_aadhar_enrolment_0_500000.csv', 'api_data_aadhar_enrolment_500000_1000000.csv', 'api_data_aadhar_enrolment_1000000_1006029.csv']
    enrol_dfs = []
    for f in enrol_files:
        try:
            enrol_dfs.append(pd.read_csv(f))
        except:
            pass
    
    if enrol_dfs:
        enrol_df = pd.concat(enrol_dfs, ignore_index=True)
        enrol_df['state_clean'] = enrol_df['state'].str.lower().str.strip().replace({'orissa': 'odisha', '100000': None})
        enrol_df = enrol_df[enrol_df['state_clean'].notna()]
        enrol_df['district_clean'] = enrol_df['district'].str.lower().str.strip().str.title()
        enrol_df['total_enrolment'] = enrol_df['age_0_5'] + enrol_df['age_5_17'] + enrol_df['age_18_greater']
        print(f"+ Enrollment: {len(enrol_df)} rows")
    else:
        enrol_df = None
        print("! Enrollment data not found")
except Exception as e:
    print(f"X Error: {e}")
    exit(1)

# Prepare data
print("\n[2/8] Preparing features...")
bio_agg = bio_df.groupby('district_clean').agg({'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'}).reset_index()
demo_agg = demo_df.groupby('district_clean').agg({'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'}).reset_index()
df = pd.merge(bio_agg, demo_agg, on='district_clean', how='outer').fillna(0)

if enrol_df is not None:
    enrol_agg = enrol_df.groupby('district_clean').agg({'total_enrolment': 'sum', 'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'}).reset_index()
    df = pd.merge(df, enrol_agg, on='district_clean', how='outer').fillna(0)

df['total_bio'] = df['bio_age_5_17'] + df['bio_age_17_']
df['total_demo'] = df['demo_age_5_17'] + df['demo_age_17_']
df['child_ratio'] = df['bio_age_5_17'] / (df['total_bio'] + 1)
df['adult_ratio'] = df['bio_age_17_'] / (df['total_bio'] + 1)
df['bio_demo_ratio'] = df['total_bio'] / (df['total_demo'] + 1)

if enrol_df is not None:
    df['enrolment_ratio'] = df['total_enrolment'] / (df['total_demo'] + 1)
    df['child_enrol_ratio'] = df['age_0_5'] / (df['total_enrolment'] + 1)
    feature_cols = ['total_bio', 'total_demo', 'child_ratio', 'adult_ratio', 'bio_demo_ratio', 'enrolment_ratio', 'child_enrol_ratio']
else:
    feature_cols = ['total_bio', 'total_demo', 'child_ratio', 'adult_ratio', 'bio_demo_ratio']

df['risk_score'] = 1 - df['bio_demo_ratio']
df['high_risk'] = (df['risk_score'] > df['risk_score'].median()).astype(int)

print(f"+ Features: {len(feature_cols)}")
print(f"+ Districts: {len(df)}")
print(f"+ High risk: {df['high_risk'].sum()} ({df['high_risk'].mean():.1%})")

# EDA Visualizations
print("\n[3/8] Generating EDA visualizations...")

# 1. Data Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')

for i, col in enumerate(feature_cols[:6]):
    row, col_idx = i // 3, i % 3
    axes[row, col_idx].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[row, col_idx].set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
    axes[row, col_idx].set_xlabel('Value')
    axes[row, col_idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('ml_analysis_results/01_data_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Correlation Matrix
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[feature_cols + ['high_risk']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.3f')
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.savefig('ml_analysis_results/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Risk Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
df['high_risk'].value_counts().plot(kind='bar', ax=ax1, color=['lightgreen', 'lightcoral'])
ax1.set_title('Risk Distribution', fontweight='bold')
ax1.set_xlabel('Risk Level')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['Low Risk', 'High Risk'], rotation=0)

df['risk_score'].hist(bins=30, ax=ax2, alpha=0.7, color='orange', edgecolor='black')
ax2.set_title('Risk Score Distribution', fontweight='bold')
ax2.set_xlabel('Risk Score')
ax2.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('ml_analysis_results/03_risk_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare ML data
print("\n[4/8] Preparing ML data...")
X = df[feature_cols].values
y = df['high_risk'].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"+ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Train models individually
print("\n[5/8] Training models individually...")
results = {}
individual_charts = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Train
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    results[name] = {
        'accuracy': float(acc), 'precision': float(prec),
        'recall': float(rec), 'f1_score': float(f1),
        'model': model, 'y_test_pred': y_test_pred, 'y_test_proba': y_test_proba
    }
    
    print(f"Val - Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")
    
    # Individual model chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{name} - Detailed Analysis', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
    ax1.set_title('Confusion Matrix (Test Set)')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax3.bar(range(len(feature_cols)), importances[indices], color='steelblue', alpha=0.8)
        ax3.set_title('Feature Importance')
        ax3.set_xticks(range(len(feature_cols)))
        ax3.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Feature Importance')
    
    # Performance Metrics
    metrics_data = [acc, prec, rec, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bars = ax4.bar(metric_names, metrics_data, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], alpha=0.8)
    ax4.set_title('Performance Metrics')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    for bar, val in zip(bars, metrics_data):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'ml_analysis_results/model_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Comparative Analysis
print("\n[6/8] Generating comparative analysis...")

# Model Performance Comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 15))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

model_names = list(results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
metric_data = {metric: [results[name][metric] for name in model_names] for metric in metrics}

colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
for i, (metric, ax) in enumerate(zip(metrics, [ax1, ax2, ax3, ax4])):
    bars = ax.bar(model_names, metric_data[metric], color=colors[i], alpha=0.8)
    ax.set_title(f'{metric.replace("_", " ").title()} Scores', fontweight='bold')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, metric_data[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('ml_analysis_results/04_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curves Comparison
print("\n[7/8] Generating ROC comparison...")
fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

for i, (name, result) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, result['y_test_proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.savefig('ml_analysis_results/05_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Best Model Analysis
print("\n[8/8] Final analysis...")
best_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_name]['model']

print(f"\nBEST MODEL: {best_name}")
print(f"F1-Score: {results[best_name]['f1_score']:.3f}")

# Final test evaluation
y_test_pred_best = results[best_name]['y_test_pred']
test_acc = accuracy_score(y_test, y_test_pred_best)
test_prec = precision_score(y_test, y_test_pred_best, zero_division=0)
test_rec = recall_score(y_test, y_test_pred_best, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred_best, zero_division=0)

print(f"\nFinal Test Performance:")
print(f"Accuracy: {test_acc:.3f}")
print(f"Precision: {test_prec:.3f}")
print(f"Recall: {test_rec:.3f}")
print(f"F1-Score: {test_f1:.3f}")

# Save models and results
with open('ml_analysis_results/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('ml_analysis_results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

results_summary = {name: {k: v for k, v in res.items() if k != 'model' and not k.startswith('y_')} for name, res in results.items()}
with open('ml_analysis_results/model_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n+ Analysis complete! All results saved to 'ml_analysis_results' folder")
print(f"+ Generated {len(models) + 5} visualization files")
print(f"+ Best model and scaler saved")