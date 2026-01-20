"""
ML Model Training for UIDAI Risk Prediction
Trains and compares multiple algorithms
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import json
from src.pipeline import UIDaiPipeline

print("=" * 70)
print("  UIDAI RISK PREDICTION - ML MODEL TRAINING")
print("=" * 70)

# Load data
print("\n[1/6] Loading data...")
pipeline = UIDaiPipeline()
df = pipeline.run()

# Prepare features
print("[2/6] Preparing features...")
print(f"Available columns: {df.columns.tolist()}")

# Use available columns from pipeline output
feature_cols = ['Risk Score']

X = df[feature_cols].fillna(0)
y = (df['Risk Level'] == 'High').astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"High risk ratio: {y.mean():.2%}")

# Define models
print("\n[3/6] Initializing algorithms...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
}

# Train and evaluate
print("\n[4/6] Training models...")
results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Train
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }
    
    print(f"    Accuracy: {accuracy:.3f} | F1: {f1:.3f} | CV F1: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# Find best model
print("\n[5/6] Comparing models...")
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']

print(f"\n{'='*70}")
print(f"  BEST MODEL: {best_model_name}")
print(f"{'='*70}")

# Detailed evaluation of best model
print("\n[6/6] Detailed evaluation of best model...")

if best_model_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Low/Medium Risk', 'High Risk']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# Feature importance
print("\nFeature Importance:")
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    importances = None

if importances is not None:
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat:30s}: {imp:.4f}")

# Save best model
print("\n[SAVING] Saving best model...")
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save results
results_summary = {
    name: {
        'accuracy': res['accuracy'],
        'precision': res['precision'],
        'recall': res['recall'],
        'f1_score': res['f1_score'],
        'cv_mean': res['cv_mean'],
        'cv_std': res['cv_std']
    }
    for name, res in results.items()
}

with open('model_comparison.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✅ Best model saved: best_model.pkl")
print(f"✅ Scaler saved: scaler.pkl")
print(f"✅ Results saved: model_comparison.json")

# Summary table
print("\n" + "="*70)
print("  MODEL COMPARISON SUMMARY")
print("="*70)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*70)
for name, res in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
    print(f"{name:<25} {res['accuracy']:>10.3f} {res['precision']:>10.3f} {res['recall']:>10.3f} {res['f1_score']:>10.3f}")
print("="*70)

print(f"\n🏆 Winner: {best_model_name} with F1-Score: {results[best_model_name]['f1_score']:.3f}")
print("\n✅ Training complete!")
