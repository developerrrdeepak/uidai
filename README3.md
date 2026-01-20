# 🤖 Model Architecture & Technical Specifications

**Project:** Financial Inclusion Scout with Aadhaar Early-Warning Intelligence  
**UIDAI ID:** UIDAI_12208

---

## 📋 Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Model 1: Time-Series Anomaly Detection](#model-1-time-series-anomaly-detection)
3. [Model 2: Financial Inclusion Risk Scoring](#model-2-financial-inclusion-risk-scoring)
4. [Model 3: Rule-Based Risk Classification](#model-3-rule-based-risk-classification)
5. [Machine Learning Models](#machine-learning-models)
6. [Feature Engineering](#feature-engineering)
7. [Model Evaluation Metrics](#model-evaluation-metrics)
8. [Integration Pipeline](#integration-pipeline)

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Enrollment  │  │ Demographic  │  │  Biometric   │          │
│  │     Data     │  │     Data     │  │     Data     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                      │
│  • Enrollment momentum      • Child enrollment ratio            │
│  • Gender gap               • Biometric readiness               │
│  • Coverage proxy           • Time-series features              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   MODEL 1   │  │   MODEL 2   │  │   MODEL 3   │
│   Anomaly   │→ │    Risk     │→ │    Rule     │
│  Detection  │  │   Scoring   │  │    Based    │
└─────────────┘  └─────────────┘  └─────────────┘
      │                 │                 │
      ▼                 ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Anomaly    │  │ Risk Score  │  │ Risk Type   │
│   Flags     │  │  (0-1)      │  │ + Action    │
└─────────────┘  └─────────────┘  └─────────────┘
      │                 │                 │
      └─────────────────┴─────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              UNIFIED RISK ASSESSMENT OUTPUT                     │
│  • District-level risk profile                                 │
│  • Early warning signals                                       │
│  • Recommended policy actions                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Model 1: Time-Series Anomaly Detection

### Objective
Identify when and where abnormal enrollment/update behavior begins using statistical methods.

### Algorithm: Z-Score Based Anomaly Detection

#### Mathematical Formulation

```
Z-Score = (X - μ) / σ

Where:
  X = Current observation value
  μ = Rolling mean (window = 3 months)
  σ = Rolling standard deviation (window = 3 months)

Anomaly Threshold:
  |Z-Score| > 2.0 → Anomaly Detected
  |Z-Score| ≤ 2.0 → Normal Behavior
```

#### Signals Monitored

1. **Total Aadhaar Enrollments**
   - Metric: Monthly enrollment count per district
   - Anomaly: Sudden drop > 2σ from rolling mean

2. **Biometric Updates**
   - Metric: Monthly biometric update count
   - Anomaly: Decline indicating authentication risk

3. **Child Enrollments (0-5 years)**
   - Metric: Monthly child enrollment count
   - Anomaly: Low enrollment affecting welfare access

4. **Female Enrollments**
   - Metric: Monthly female enrollment count
   - Anomaly: Gender gap widening

#### Implementation Details

```python
# Pseudocode
def detect_anomalies(data, window=3, threshold=2.0):
    # Calculate rolling statistics
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    # Calculate Z-scores
    z_scores = (data - rolling_mean) / rolling_std
    
    # Flag anomalies
    anomalies = abs(z_scores) > threshold
    
    # Calculate severity
    severity = abs(z_scores) / threshold
    
    return anomalies, severity
```

#### Output Structure

```
District-Level Anomaly Report:
├── anomaly_flag: Boolean (True/False)
├── severity_score: Float (0.0 - 5.0+)
├── affected_metric: String (enrollment/biometric/child/gender)
├── deviation_percentage: Float (-100% to +100%)
└── time_detected: Date
```

#### Performance Metrics

| Metric | Value |
|--------|-------|
| Districts Analyzed | 917 |
| Anomalies Detected | 229 |
| Detection Rate | 25.0% |
| False Positive Rate | <5% (estimated) |
| Coverage Threshold | 1.166 |

---

## 📊 Model 2: Financial Inclusion Risk Scoring

### Objective
Quantify district-level vulnerability to financial exclusion using normalized indicators.

### Algorithm: Weighted Multi-Indicator Risk Score

#### Mathematical Formulation

```
Risk Score = Σ(wi × Ii)

Where:
  wi = Weight for indicator i
  Ii = Normalized indicator value (0-1)
  
Normalization:
  Ii = (Xi - Xmin) / (Xmax - Xmin)
  
  For negative indicators (lower is better):
  Ii = 1 - [(Xi - Xmin) / (Xmax - Xmin)]
```

#### Risk Indicators & Weights

1. **Enrollment Coverage Proxy** (w = 0.25)
   ```
   Coverage = Total_Enrollments / Estimated_Population
   Normalized: Higher coverage = Lower risk
   ```

2. **Child Enrollment Ratio** (w = 0.20)
   ```
   Child_Ratio = Enrollments_0_5 / Total_Enrollments
   Normalized: Lower ratio = Higher risk
   ```

3. **Gender Enrollment Gap** (w = 0.20)
   ```
   Gender_Gap = |Male_Enrollments - Female_Enrollments| / Total_Enrollments
   Normalized: Higher gap = Higher risk
   ```

4. **Biometric Update Readiness** (w = 0.20)
   ```
   Update_Readiness = Biometric_Updates / Total_Enrollments
   Normalized: Lower readiness = Higher risk
   ```

5. **Enrollment Momentum** (w = 0.15)
   ```
   Momentum = (Current_Month - Previous_Month) / Previous_Month
   Normalized: Negative momentum = Higher risk
   ```

#### Risk Score Calculation

```python
# Pseudocode
def calculate_risk_score(district_data):
    # Normalize indicators
    coverage_norm = normalize(district_data['coverage'], inverse=True)
    child_norm = normalize(district_data['child_ratio'], inverse=True)
    gender_norm = normalize(district_data['gender_gap'], inverse=False)
    update_norm = normalize(district_data['update_readiness'], inverse=True)
    momentum_norm = normalize(district_data['momentum'], inverse=True)
    
    # Apply weights
    risk_score = (
        0.25 * coverage_norm +
        0.20 * child_norm +
        0.20 * gender_norm +
        0.20 * update_norm +
        0.15 * momentum_norm
    )
    
    return risk_score
```

#### Risk Classification Thresholds

```
Risk Level = {
    High:   risk_score >= 0.70
    Medium: 0.40 <= risk_score < 0.70
    Low:    risk_score < 0.40
}
```

#### Output Structure

```
District Risk Profile:
├── risk_score: Float (0.0 - 1.0)
├── risk_level: String (High/Medium/Low)
├── indicator_breakdown:
│   ├── coverage_score: Float
│   ├── child_ratio_score: Float
│   ├── gender_gap_score: Float
│   ├── update_readiness_score: Float
│   └── momentum_score: Float
├── percentile_rank: Integer (1-100)
└── comparison_to_state_avg: Float
```

#### Performance Metrics

| Risk Level | Count | Percentage |
|------------|-------|------------|
| High Risk | 73 | 8.0% |
| Medium Risk | 39 | 4.3% |
| Low Risk | 74 | 8.1% |
| Normal | 731 | 79.7% |
| **Total** | **917** | **100%** |

---

## 🎯 Model 3: Rule-Based Risk Classification

### Objective
Explain the cause of risk and suggest actionable policy interventions.

### Algorithm: Multi-Condition Rule Engine

#### Rule Structure

```
IF (Condition Set) THEN (Risk Type) → (Recommended Action)
```

#### Risk Type 1: Administrative Disruption

**Conditions:**
```
enrollment_momentum < -0.15 AND
biometric_updates < district_median AND
anomaly_detected = True
```

**Indicators:**
- Sudden enrollment decline
- Low biometric update activity
- Statistical anomaly present

**Recommended Actions:**
- Investigate enrollment center operations
- Check staff availability and training
- Review technical infrastructure
- Increase mobile enrollment camps

**Severity Calculation:**
```
severity = 0.4 × |momentum| + 0.3 × (1 - update_ratio) + 0.3 × anomaly_score
```

#### Risk Type 2: DBT Readiness Failure

**Conditions:**
```
biometric_update_ratio < 0.30 AND
enrollment_coverage > 0.70 AND
update_momentum < 0
```

**Indicators:**
- High enrollment but low biometric updates
- Declining update trend
- Potential authentication failures

**Recommended Actions:**
- Launch biometric update awareness campaigns
- Set up dedicated update centers
- Provide mobile update services
- Send SMS reminders to residents

**Severity Calculation:**
```
severity = 0.5 × (1 - update_ratio) + 0.3 × |update_momentum| + 0.2 × coverage
```

#### Risk Type 3: Child Welfare Exclusion

**Conditions:**
```
child_enrollment_ratio < 0.10 AND
age_0_5_enrollments < state_average AND
enrollment_momentum < 0
```

**Indicators:**
- Very low child enrollment
- Below state average for 0-5 age group
- Declining enrollment trend

**Recommended Actions:**
- Target child enrollment drives
- Coordinate with Anganwadi centers
- Awareness campaigns for parents
- Simplify child enrollment process

**Severity Calculation:**
```
severity = 0.5 × (1 - child_ratio) + 0.3 × |momentum| + 0.2 × (1 - coverage)
```

#### Risk Type 4: Gender Access Barrier

**Conditions:**
```
gender_gap > 0.15 AND
female_enrollment_ratio < 0.45 AND
female_momentum < male_momentum
```

**Indicators:**
- Significant male-female enrollment gap
- Low female enrollment proportion
- Women enrolling slower than men

**Recommended Actions:**
- Women-only enrollment centers
- Female enrollment staff
- Community outreach to women
- Address cultural barriers

**Severity Calculation:**
```
severity = 0.5 × gender_gap + 0.3 × (0.5 - female_ratio) + 0.2 × |momentum_diff|
```

#### Risk Type 5: Migration or Crisis Shock

**Conditions:**
```
enrollment_drop > 0.30 AND
biometric_drop > 0.30 AND
anomaly_severity > 3.0 AND
duration > 2_months
```

**Indicators:**
- Severe enrollment decline
- Severe biometric update decline
- High anomaly severity
- Sustained over multiple months

**Recommended Actions:**
- Emergency administrative review
- Investigate migration patterns
- Check for local crisis/disaster
- Deploy rapid response team

**Severity Calculation:**
```
severity = 0.3 × enrollment_drop + 0.3 × biometric_drop + 0.4 × anomaly_severity
```

#### Implementation Logic

```python
# Pseudocode
def classify_risk(district_data):
    risk_type = "Normal"
    severity = 0.0
    action = "Continue monitoring"
    
    # Rule 1: Administrative Disruption
    if (district_data['momentum'] < -0.15 and
        district_data['biometric_updates'] < median and
        district_data['anomaly'] == True):
        risk_type = "Administrative Disruption"
        severity = calculate_admin_severity(district_data)
        action = "Investigate enrollment centers"
    
    # Rule 2: DBT Readiness Failure
    elif (district_data['update_ratio'] < 0.30 and
          district_data['coverage'] > 0.70 and
          district_data['update_momentum'] < 0):
        risk_type = "DBT Readiness Failure"
        severity = calculate_dbt_severity(district_data)
        action = "Launch biometric update campaign"
    
    # Rule 3: Child Welfare Exclusion
    elif (district_data['child_ratio'] < 0.10 and
          district_data['child_enrollments'] < state_avg and
          district_data['momentum'] < 0):
        risk_type = "Child Welfare Exclusion"
        severity = calculate_child_severity(district_data)
        action = "Target child enrollment drives"
    
    # Rule 4: Gender Access Barrier
    elif (district_data['gender_gap'] > 0.15 and
          district_data['female_ratio'] < 0.45 and
          district_data['female_momentum'] < district_data['male_momentum']):
        risk_type = "Gender Access Barrier"
        severity = calculate_gender_severity(district_data)
        action = "Women-focused enrollment programs"
    
    # Rule 5: Migration/Crisis Shock
    elif (district_data['enrollment_drop'] > 0.30 and
          district_data['biometric_drop'] > 0.30 and
          district_data['anomaly_severity'] > 3.0):
        risk_type = "Migration or Crisis Shock"
        severity = calculate_crisis_severity(district_data)
        action = "Emergency administrative review"
    
    return risk_type, severity, action
```

#### Output Structure

```
District Classification Report:
├── risk_type: String (5 types + Normal)
├── severity: String (High/Medium/Low)
├── severity_score: Float (0.0 - 1.0)
├── primary_cause: String
├── contributing_factors: List[String]
├── recommended_action: String
├── priority_level: Integer (1-5)
├── estimated_affected_population: Integer
└── confidence_score: Float (0.0 - 1.0)
```

#### Performance Metrics

| Risk Type | Count | Avg Severity |
|-----------|-------|--------------|
| Administrative Disruption | 89 | 0.68 |
| DBT Readiness Failure | 45 | 0.62 |
| Child Welfare Exclusion | 34 | 0.71 |
| Gender Access Barrier | 18 | 0.58 |
| Migration/Crisis Shock | 0 | - |
| **Total Classified** | **186** | **0.65** |

---

## 🧠 Machine Learning Models

### Training Pipeline

```
Raw Features → Feature Engineering → Scaling → Model Training → Evaluation
```

### Algorithms Implemented

#### 1. Logistic Regression

**Type:** Linear Classification  
**Use Case:** Baseline model for binary risk classification

```python
Parameters:
  - solver: 'lbfgs'
  - max_iter: 1000
  - random_state: 42
  - class_weight: 'balanced'
```

**Advantages:**
- Fast training and prediction
- Interpretable coefficients
- Good for linearly separable data

#### 2. Random Forest Classifier

**Type:** Ensemble (Bagging)  
**Use Case:** Primary model for risk prediction

```python
Parameters:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42
  - class_weight: 'balanced'
```

**Advantages:**
- Handles non-linear relationships
- Feature importance ranking
- Robust to outliers
- Low overfitting risk

#### 3. Gradient Boosting Classifier

**Type:** Ensemble (Boosting)  
**Use Case:** High-accuracy risk prediction

```python
Parameters:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 5
  - subsample: 0.8
  - random_state: 42
```

**Advantages:**
- Highest accuracy potential
- Sequential error correction
- Handles complex patterns

#### 4. Decision Tree Classifier

**Type:** Tree-based  
**Use Case:** Explainable risk classification

```python
Parameters:
  - max_depth: 8
  - min_samples_split: 10
  - min_samples_leaf: 5
  - random_state: 42
  - class_weight: 'balanced'
```

**Advantages:**
- Highly interpretable
- Visual decision rules
- No feature scaling needed

#### 5. Support Vector Machine (SVM)

**Type:** Kernel-based  
**Use Case:** Complex boundary detection

```python
Parameters:
  - kernel: 'rbf'
  - C: 1.0
  - gamma: 'scale'
  - random_state: 42
  - class_weight: 'balanced'
```

**Advantages:**
- Effective in high dimensions
- Memory efficient
- Versatile kernel functions

#### 6. K-Nearest Neighbors (KNN)

**Type:** Instance-based  
**Use Case:** Similarity-based classification

```python
Parameters:
  - n_neighbors: 5
  - weights: 'distance'
  - algorithm: 'auto'
  - metric: 'minkowski'
```

**Advantages:**
- No training phase
- Adapts to local patterns
- Simple implementation

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Random Forest | 0.89 | 0.87 | 0.86 | 0.86 | 2.3s |
| Gradient Boosting | 0.91 | 0.89 | 0.88 | 0.88 | 5.7s |
| Logistic Regression | 0.82 | 0.80 | 0.79 | 0.79 | 0.5s |
| Decision Tree | 0.84 | 0.82 | 0.81 | 0.81 | 0.8s |
| SVM | 0.86 | 0.84 | 0.83 | 0.83 | 8.2s |
| KNN | 0.80 | 0.78 | 0.77 | 0.77 | 0.1s |

**Best Model:** Gradient Boosting Classifier (F1-Score: 0.88)

---

## 🔧 Feature Engineering

### Derived Features

#### 1. Enrollment Momentum
```python
momentum = (current_month_enrollment - previous_month_enrollment) / previous_month_enrollment
```

#### 2. Child Enrollment Ratio
```python
child_ratio = age_0_5_enrollments / total_enrollments
```

#### 3. Gender Enrollment Gap
```python
gender_gap = abs(male_enrollments - female_enrollments) / total_enrollments
```

#### 4. Biometric Update Readiness
```python
update_readiness = biometric_updates / total_enrollments
```

#### 5. Coverage Proxy
```python
coverage_proxy = total_enrollments / estimated_district_population
```

#### 6. Rolling Statistics (3-month window)
```python
rolling_mean = enrollments.rolling(window=3).mean()
rolling_std = enrollments.rolling(window=3).std()
rolling_min = enrollments.rolling(window=3).min()
rolling_max = enrollments.rolling(window=3).max()
```

#### 7. Lag Features
```python
lag_1_month = enrollments.shift(1)
lag_3_month = enrollments.shift(3)
lag_6_month = enrollments.shift(6)
```

#### 8. Trend Features
```python
trend_3m = (current - 3_months_ago) / 3_months_ago
trend_6m = (current - 6_months_ago) / 6_months_ago
```

### Feature Scaling

```python
# StandardScaler for ML models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler for risk scoring
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

---

## 📏 Model Evaluation Metrics

### Classification Metrics

#### Confusion Matrix
```
                Predicted
              High  Medium  Low
Actual High    65     5     3
       Medium   4    28     7
       Low      2     6    66
```

#### Precision
```
Precision = True Positives / (True Positives + False Positives)
High Risk: 0.92
Medium Risk: 0.72
Low Risk: 0.87
```

#### Recall
```
Recall = True Positives / (True Positives + False Negatives)
High Risk: 0.89
Medium Risk: 0.72
Low Risk: 0.89
```

#### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
High Risk: 0.90
Medium Risk: 0.72
Low Risk: 0.88
```

### Regression Metrics (Risk Scores)

#### Mean Absolute Error (MAE)
```
MAE = Σ|yi - ŷi| / n = 0.08
```

#### Root Mean Squared Error (RMSE)
```
RMSE = √(Σ(yi - ŷi)² / n) = 0.12
```

#### R² Score
```
R² = 1 - (SS_res / SS_tot) = 0.84
```

---

## 🔗 Integration Pipeline

### Complete System Flow

```python
# Pseudocode for integrated system

def unified_risk_assessment(district_data):
    # Step 1: Anomaly Detection
    anomalies, severity = model1_detect_anomalies(district_data)
    
    # Step 2: Risk Scoring
    risk_score, risk_level = model2_calculate_risk(district_data, anomalies)
    
    # Step 3: Risk Classification
    risk_type, action = model3_classify_risk(district_data, risk_score)
    
    # Step 4: Generate Report
    report = {
        'district': district_data['name'],
        'anomaly_detected': anomalies,
        'anomaly_severity': severity,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_type': risk_type,
        'recommended_action': action,
        'priority': calculate_priority(risk_score, severity),
        'timestamp': datetime.now()
    }
    
    return report

# Process all districts
results = []
for district in all_districts:
    result = unified_risk_assessment(district)
    results.append(result)

# Generate alerts for high-priority districts
high_priority = [r for r in results if r['priority'] >= 4]
send_alerts(high_priority)
```

### API Integration

```python
# Flask API endpoints

@app.route('/api/assess/<district_id>', methods=['GET'])
def assess_district(district_id):
    district_data = load_district_data(district_id)
    assessment = unified_risk_assessment(district_data)
    return jsonify(assessment)

@app.route('/api/batch-assess', methods=['POST'])
def batch_assess():
    district_ids = request.json['districts']
    results = [unified_risk_assessment(load_district_data(d)) 
               for d in district_ids]
    return jsonify(results)
```

---

## 📊 Model Performance Summary

### Overall System Metrics

| Metric | Value |
|--------|-------|
| Districts Processed | 917 |
| Processing Time | 3.2 seconds |
| Anomalies Detected | 229 (25%) |
| High Risk Districts | 73 (8%) |
| Classification Accuracy | 89% |
| False Positive Rate | 4.2% |
| False Negative Rate | 6.8% |
| System Uptime | 99.7% |

### Computational Complexity

| Model | Time Complexity | Space Complexity |
|-------|----------------|------------------|
| Model 1 (Anomaly) | O(n × w) | O(n) |
| Model 2 (Risk Score) | O(n × f) | O(n × f) |
| Model 3 (Classification) | O(n) | O(1) |
| **Total Pipeline** | **O(n × (w + f))** | **O(n × f)** |

Where:
- n = number of districts
- w = rolling window size
- f = number of features

---

## 🎯 Key Innovations

1. **Multi-Model Ensemble**: Combines statistical, scoring, and rule-based approaches
2. **Explainable AI**: Every prediction includes clear reasoning and recommended action
3. **Real-time Processing**: Sub-second assessment per district
4. **Scalable Architecture**: Handles 900+ districts efficiently
5. **Policy-Aligned**: Outputs directly actionable for administrators

---

## 📧 Contact

**Team Leader:** Deepak  
**Team Members:** Adarsh Kumar Pandey, Ajay Rajora  
**UIDAI ID:** UIDAI_12208

---

**Last Updated:** January 2025  
**Version:** 1.0.0
