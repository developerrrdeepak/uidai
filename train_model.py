import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("  UIDAI ML MODEL TRAINING - Using Cleaned CSV Data")
print("="*70)

# Load cleaned data
print("\n[1/5] Loading cleaned CSV files...")
try:
    bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
    demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')
    print(f"✓ Biometric data: {len(bio_df)} rows")
    print(f"✓ Demographic data: {len(demo_df)} rows")
    
    # Load enrollment data
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
        print(f"✓ Enrollment data: {len(enrol_df)} rows")
    else:
        enrol_df = None
        print("⚠️ Enrollment data not found, continuing without it")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# Aggregate by district
print("\n[2/5] Aggregating data by district...")
bio_agg = bio_df.groupby('district_clean').agg({
    'bio_age_5_17': 'sum',
    'bio_age_17_': 'sum'
}).reset_index()

demo_agg = demo_df.groupby('district_clean').agg({
    'demo_age_5_17': 'sum',
    'demo_age_17_': 'sum'
}).reset_index()

# Merge datasets
df = pd.merge(bio_agg, demo_agg, on='district_clean', how='outer').fillna(0)

# Merge enrollment if available
if enrol_df is not None:
    enrol_agg = enrol_df.groupby('district_clean').agg({'total_enrolment': 'sum', 'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'}).reset_index()
    df = pd.merge(df, enrol_agg, on='district_clean', how='outer').fillna(0)

# Create features
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

# Create risk label (districts with low biometric coverage)
df['risk_score'] = 1 - df['bio_demo_ratio']
df['high_risk'] = (df['risk_score'] > df['risk_score'].median()).astype(int)

print(f"✓ Processed {len(df)} districts")
print(f"✓ High risk districts: {df['high_risk'].sum()} ({df['high_risk'].mean():.1%})")

# Prepare features
print("\n[3/5] Preparing features...")
X = df[feature_cols].values
y = df['high_risk'].values

# Split data into train/validation/test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Features: {len(feature_cols)}")

# Train models using validation set for evaluation
print("\n[4/6] Training algorithms..."))
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    # Train on training set
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)  # Evaluate on validation set
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)  # Evaluate on validation set
    
    # Evaluate on validation set (not test set!)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    results[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'model': model
    }
    
    print(f"  {name:25s} - Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

# Select best model
print("\n[5/6] Selecting best model..."))
best_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_name]['model']

print(f"\n{'='*70}")
print(f"  🏆 BEST MODEL: {best_name} (based on validation performance)")
print(f"{'='*70}")
print(f"  Validation Performance:")
print(f"  Accuracy:  {results[best_name]['accuracy']:.3f}")
print(f"  Precision: {results[best_name]['precision']:.3f}")
print(f"  Recall:    {results[best_name]['recall']:.3f}")
print(f"  F1-Score:  {results[best_name]['f1_score']:.3f}")

# Detailed evaluation on unseen test set
print(f"\n{'='*70}")
print("  FINAL EVALUATION ON UNSEEN TEST DATA")
print(f"{'='*70}")

if best_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

# Final test metrics
test_acc = accuracy_score(y_test, y_pred_best)
test_prec = precision_score(y_test, y_pred_best, zero_division=0)
test_rec = recall_score(y_test, y_pred_best, zero_division=0)
test_f1 = f1_score(y_test, y_pred_best, zero_division=0)

print(f"\n  Final Test Performance:")
print(f"  Accuracy:  {test_acc:.3f}")
print(f"  Precision: {test_prec:.3f}")
print(f"  Recall:    {test_rec:.3f}")
print(f"  F1-Score:  {test_f1:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Low Risk', 'High Risk']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"  True Negatives:  {cm[0,0]:3d}  |  False Positives: {cm[0,1]:3d}")
print(f"  False Negatives: {cm[1,0]:3d}  |  True Positives:  {cm[1,1]:3d}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importance:")
    importances = best_model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat:20s}: {imp:.4f}")

# Save models
print(f"\n{'='*70}")
print("  SAVING MODELS")
print(f"{'='*70}")

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: best_model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)
print("✓ Saved: feature_columns.json")

# Save results
results_summary = {
    name: {
        'accuracy': res['accuracy'],
        'precision': res['precision'],
        'recall': res['recall'],
        'f1_score': res['f1_score']
    }
    for name, res in results.items()
}

with open('model_comparison.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("✓ Saved: model_comparison.json")

# Summary table
print(f"\n{'='*70}")
print("  MODEL COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*70)
for name, res in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
    print(f"{name:<25} {res['accuracy']:>10.3f} {res['precision']:>10.3f} {res['recall']:>10.3f} {res['f1_score']:>10.3f}")
print("="*70)

print(f"\n✅ Training complete! Best model: {best_name}")
print(f"✅ Model evaluated on genuine unseen test data!")
print(f"✅ All files saved successfully.")

# Generate visualizations
print("\n[6/6] Generating visualizations...")

# 1. Model Performance Comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

model_names = list(results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
metric_data = {metric: [results[name][metric] for name in model_names] for metric in metrics}

# Accuracy
ax1.bar(model_names, metric_data['accuracy'], color='skyblue', alpha=0.8)
ax1.set_title('Accuracy Scores', fontweight='bold')
ax1.set_ylabel('Score')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(metric_data['accuracy']):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Precision
ax2.bar(model_names, metric_data['precision'], color='lightcoral', alpha=0.8)
ax2.set_title('Precision Scores', fontweight='bold')
ax2.set_ylabel('Score')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(metric_data['precision']):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Recall
ax3.bar(model_names, metric_data['recall'], color='lightgreen', alpha=0.8)
ax3.set_title('Recall Scores', fontweight='bold')
ax3.set_ylabel('Score')
ax3.tick_params(axis='x', rotation=45)
for i, v in enumerate(metric_data['recall']):
    ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# F1-Score
ax4.bar(model_names, metric_data['f1_score'], color='gold', alpha=0.8)
ax4.set_title('F1-Scores', fontweight='bold')
ax4.set_ylabel('Score')
ax4.tick_params(axis='x', rotation=45)
for i, v in enumerate(metric_data['f1_score']):
    ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. ROC Curves for all models
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

for i, (name, model) in enumerate(models.items()):
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=colors[i], lw=2, 
            label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.savefig('roc_curves_all_models.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Confusion Matrix for Best Model
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'], ax=ax)
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
ax.set_xlabel('Predicted Label', fontweight='bold')
plt.savefig('confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax.bar(range(len(feature_cols)), importances[indices], color='steelblue', alpha=0.8)
    ax.set_title(f'Feature Importance - {best_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontweight='bold')
    ax.set_ylabel('Importance', fontweight='bold')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right')
    
    for i, v in enumerate(importances[indices]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. Model Comparison Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Select top 4 models for clarity
top_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:4]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors = ['red', 'blue', 'green', 'orange']
for i, (name, res) in enumerate(top_models):
    values = [res[metric] for metric in metrics]
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
ax.set_ylim(0, 1)
ax.set_title('Top 4 Models - Performance Radar', size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
ax.grid(True)
plt.savefig('model_radar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Generated 5 visualization charts:")
print("  - model_performance_comparison.png")
print("  - roc_curves_all_models.png")
print("  - confusion_matrix_best_model.png")
if hasattr(best_model, 'feature_importances_'):
    print("  - feature_importance.png")
print("  - model_radar_chart.png")
