# Financial Inclusion Scout with Aadhaar Early-Warning Intelligence

**A Proactive System for Detecting Aadhaar-Based Financial Inclusion Risks**

---

## 📋 Project Information

**UIDAI ID:** UIDAI_12208

**Team Members:**
- Deepak (Team Leader)
- Adarsh Kumar Pandey
- Ajay Rajora
- Sourav kumar

---

## 📑 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset Used](#dataset-used)
3. [Preprocessing and Methodology](#preprocessing-and-methodology)
4. [Logic and Technical Implementation](#logic-and-technical-implementation)
5. [Analysis](#analysis)
6. [Visualizations](#visualizations)
7. [Source Code](#source-code)
8. [Impact](#impact)
9. [How to Run](#how-to-run)

---

## 🎯 Problem Statement

**Main Problem:** Unlocking Societal Trends in Aadhaar Enrolment and Updates

**Sub-problem:** Aadhaar-based enrollment and update gaps silently block financial inclusion and welfare delivery, but these risks are detected too late.

### Key Challenge

At present, Aadhaar monitoring is mainly based on total numbers and past reports. This system does not clearly show:
- ✗ Early signs that enrolment activity is declining
- ✗ Sudden drops in biometric update activity
- ✗ Low enrolment of children (0–5 years) or gender imbalance
- ✗ Which districts need urgent administrative attention

### Scope Clarification

This problem **does not** deal with:
- ✗ Checking or verifying individual Aadhaar identities
- ✗ Biometric authentication or matching
- ✗ Detecting fraud or misuse
- ✗ Analysing data at the individual or beneficiary level

### ✅ Solution

An explainable AI system that uses Aadhaar enrolment, demographic, and biometric update data to:
- ✓ Identify **high-risk districts**
- ✓ Detect **early warning signals**
- ✓ Recommend **corrective actions**

---

## 📊 Dataset Used

This project uses **only the Aadhaar datasets provided by UIDAI** for the hackathon.

### 1. Aadhaar Enrolment Data

**Main Columns Used:**
- State and District name
- Time period (month/year)
- Total Aadhaar enrolments
- New enrolments

**Use in Project:** Study enrollment trends, identify districts with low or declining enrollment activity, and assess overall enrollment coverage related to financial inclusion.

### 2. Aadhaar Demographic Data

**Main Columns Used:**
- State and District name
- Age group (0–5 years)
- Gender (Male/Female)
- Enrolment count

**Use in Project:** Analyze child enrolment levels and gender distribution in Aadhaar enrolment, helping to identify demographic gaps that may affect welfare access.

### 3. Aadhaar Biometric Update Data

**Main Columns Used:**
- State and District name
- Time period
- Number of biometric updates

**Use in Project:** Monitor biometric update trends and detect districts where low update activity may lead to authentication or DBT-related issues.

### Justification for Dataset Selection

Aadhaar enrolment, demographic, and biometric update data together provide a reliable and sufficient basis to assess district-level readiness for financial inclusion and welfare delivery without compromising data privacy or security.

---

## 🔧 Preprocessing and Methodology

**Pipeline:** Raw Data → Cleaning → Pandas → Final Data

### 1. Raw Data Ingestion

**Key Actions:**
- CSV files loaded using Pandas DataFrames
- Time variables converted to standardized monthly format
- State and district identifiers normalized for consistency

### 2. Data Cleaning

**Key Steps:**
- Missing values handled using rolling-window imputation
- Outliers intentionally retained (may indicate operational anomalies)
- Negative or logically invalid values removed
- Demographic categories standardized

**Files Generated:**
- `aadhaar_biometric_cleaned.csv` (917,008 rows, 907 districts)
- `aadhaar_demographic_cleaned.csv` (882,744 rows, 814 districts)
- `aadhaar_enrollment_cleaned.csv` (cleaned enrollment data)

### 3. Feature Engineering

**Derived Features:**
- Enrolment momentum (growth/decline trends)
- Child enrolment ratio (0–5 years)
- Gender enrolment gap
- Biometric update readiness ratio

### 4. Final Analytical Dataset

**Characteristics:**
- District-level aggregation
- Time-indexed
- Ready for time-series anomaly detection, risk scoring, and classification

---

## 🤖 Logic and Technical Implementation

### System Architecture

```
UIDAI Analytical Data
        ↓
Model 1: Anomaly Detection
        ↓
Model 2: Risk Scoring
        ↓
Model 3: Risk Classification
        ↓
Alerts + Policy Recommendations
        ↓
    Dashboard
```

### Model 1: Time-Series Anomaly Detection

**Objective:** Identify when abnormal behaviour begins

**Method:** Rolling Mean and Rolling Standard Deviation using Z-score technique

**Signals Monitored:**
- Total Aadhaar enrolments
- Biometric updates
- Child enrolments (0–5 years)
- Female enrolments

**Output:**
- Anomaly flag (Normal / Alert)
- Severity score indicating intensity of deviation

**Visualization:** `model1_visuals/MODEL1_COMPREHENSIVE_DASHBOARD.png`

### Model 2: Financial Inclusion Risk Scoring

**Objective:** Identify which districts are vulnerable

**Indicators Used (Normalized):**
- Enrolment coverage proxy
- Child enrolment ratio
- Gender enrolment gap
- Biometric update readiness ratio
- Enrolment momentum

**Risk Score Output:**
- Continuous risk score (0-1)
- Risk category: Low / Medium / High

**Visualization:** `model2_visuals/MODEL2_COMPREHENSIVE_DASHBOARD.png`

### Model 3: Rule-Based Risk Classification

**Objective:** Explain the cause and suggest action

**Risk Types Identified:**
- Administrative Disruption
- DBT Readiness Failure
- Child Welfare Exclusion
- Gender Access Barrier
- Migration or Crisis Shock

**Output:**
- Identified risk type
- Clear reason explanation
- Policy-aligned recommended action

**Visualization:** `model3_visuals/MODEL3_COMPREHENSIVE_DASHBOARD.png`

---

## 📈 Analysis

### 1. Univariate Analysis

**Analysis Performed:**
- Distribution of total Aadhaar enrolments across districts
- Trend of biometric updates over time
- Distribution of child (0–5 years) enrolments
- Distribution of female enrolments

**Key Observations:**
- Some districts show stable enrolment but declining update activity
- Child enrolment levels vary significantly across districts
- Gender-based enrolment imbalance exists in specific regions

### 2. Bivariate Analysis

**Analysis Performed:**
- Enrolment trend vs biometric update trend
- Gender enrolment vs biometric update readiness
- Child enrolment ratio vs overall enrolment momentum

**Key Observations:**
- Districts with declining biometric updates show higher risk of service disruption
- Gender enrolment gaps associated with lower update readiness
- Low child enrolment + slow growth indicates future welfare exclusion

### 3. Trivariate Analysis

**Analysis Performed:**
- Gender gap + child enrolment ratio + anomaly frequency
- Enrolment momentum + biometric update trend + demographic imbalance

**Key Observations:**
- Districts with anomalies across multiple indicators at higher systemic risk
- Combined demographic and operational weaknesses indicate early-stage exclusion
- These risks not visible when indicators viewed separately

---

## 📊 Visualizations

### Model 1: Anomaly Detection Dashboard

![Model 1 Dashboard](model1_visuals/MODEL1_COMPREHENSIVE_DASHBOARD.png)

**Features:**
- KPI cards showing total districts, anomalies detected, normal districts, and anomaly rate
- Coverage distribution histogram with anomaly threshold
- Scatter plot: Biometric vs Demographic coverage
- Top 10 anomalous districts
- Detailed statistics

**Results:**
- **917 districts** analyzed
- **229 anomalies** detected (25% detection rate)
- Coverage threshold: 1.166

### Model 2: Risk Scoring Dashboard

![Model 2 Dashboard](model2_visuals/MODEL2_COMPREHENSIVE_DASHBOARD.png)

**Features:**
- Risk level KPI cards (High/Medium/Low)
- Risk score distribution by level
- Donut chart showing risk level proportions
- Top 12 high-risk districts (color-coded)
- Risk scoring statistics

**Results:**
- **917 districts** scored
- **73 High Risk** districts
- **39 Medium Risk** districts
- **74 Low Risk** districts

### Model 3: Rule-Based Classification Dashboard

![Model 3 Dashboard](model3_visuals/MODEL3_COMPREHENSIVE_DASHBOARD.png)

**Features:**
- Severity KPI cards (High/Medium/Low)
- Risk type distribution bar chart
- Stacked bar: Severity by risk type
- Top 10 high-risk districts (color = risk type)
- Classification statistics and rule logic

**Results:**
- **917 districts** classified
- **4 risk types** identified
- **73 High Severity** districts
- **52 Medium Severity** districts
- **61 Low Severity** districts

### Interactive React Dashboard

![React Dashboard](screenshots/react_dashboard.png)

**Access:** Open `advanced_dashboard.html` in browser (requires Flask API running)

**Features:**
- 4 interactive tabs: Overview, All Districts, Analytics, High Risk Alerts
- Real-time stats and filtering
- Search functionality
- Export to CSV
- Pagination (20 items per page)
- Modern dark theme with glassmorphism

---

## 💻 Source Code

### Project Structure

```
uidai data/
├── Data Cleaning Scripts
│   ├── analysis.py                    # Enrollment data cleaning
│   ├── analysis2.py                   # Demographic data cleaning
│   └── analysis3.py                   # Biometric data cleaning
│
├── Cleaned Data Files
│   ├── aadhaar_biometric_cleaned.csv  # 917,008 rows, 907 districts
│   ├── aadhaar_demographic_cleaned.csv # 882,744 rows, 814 districts
│   └── aadhaar_enrollment_cleaned.csv  # Cleaned enrollment data
│
├── Model Training & Execution
│   ├── train_model.py                 # ML model training (6 algorithms)
│   ├── rule_based_risk_classification_model3.py # Model 3 implementation
│   └── complete_unified_system.py     # All 3 models integrated
│
├── Visualization Generators
│   ├── model1_single_chart.py         # Model 1 comprehensive dashboard
│   ├── model2_single_chart.py         # Model 2 comprehensive dashboard
│   └── model3_single_chart.py         # Model 3 comprehensive dashboard
│
├── Dashboard & API
│   ├── api.py                         # Flask REST API (port 5001)
│   ├── advanced_dashboard.html        # React TypeScript dashboard
│   └── app.py                         # Streamlit dashboard (alternative)
│
├── Generated Visuals
│   ├── model1_visuals/                # Model 1 charts
│   ├── model2_visuals/                # Model 2 charts
│   └── model3_visuals/                # Model 3 charts
│
└── Documentation
    ├── README.md                      # This file
    ├── CLEANED_FILES_VERIFICATION.md  # Data verification
    └── ML_TRAINING_GUIDE.md           # ML training guide
```

### Key Scripts

#### 1. Data Cleaning
```bash
python analysis.py    # Clean enrollment data
python analysis2.py   # Clean demographic data
python analysis3.py   # Clean biometric data
```

#### 2. Model Training
```bash
python train_model.py  # Train 6 ML algorithms
```

#### 3. Generate Visualizations
```bash
python model1_single_chart.py  # Generate Model 1 dashboard
python model2_single_chart.py  # Generate Model 2 dashboard
python model3_single_chart.py  # Generate Model 3 dashboard
```

#### 4. Run Dashboard
```bash
# Start Flask API
python api.py

# Open advanced_dashboard.html in browser
```

### Technologies Used

**Data Processing:**
- Python 3.11
- Pandas, NumPy
- Scikit-learn

**Machine Learning:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- Decision Tree
- SVM
- K-Nearest Neighbors

**Visualization:**
- Matplotlib
- Seaborn
- Chart.js (React dashboard)

**Dashboard:**
- React 18 (TypeScript via Babel)
- Flask (REST API)
- Streamlit (alternative)

---

## 🎯 Impact

### Problem Impact

Delayed detection of Aadhaar enrolment and update gaps leads to:
- ✗ Financial exclusion
- ✗ DBT failures
- ✗ Denial of welfare services

These issues affect vulnerable populations: children, women, migrants, and rural citizens.

### Solution Impact

#### Immediate Impact
- ✓ Early identification of districts at risk of Aadhaar-linked exclusion
- ✓ Timely administrative intervention before service disruption
- ✓ Reduction in authentication and DBT failures

#### Operational Impact
- ✓ Better planning of enrolment and biometric update drives
- ✓ District-level prioritization based on risk instead of assumptions
- ✓ Improved monitoring of child and gender inclusion

#### Policy Impact
- ✓ Data-driven governance using existing UIDAI datasets
- ✓ Shift from reactive reporting to preventive decision-making
- ✓ Scalable monitoring framework applicable at state and national levels

### Quantified Results

| Metric | Value |
|--------|-------|
| Districts Analyzed | 917 |
| Anomalies Detected | 229 (25%) |
| High Risk Districts | 73 |
| Medium Risk Districts | 39 |
| Low Risk Districts | 74 |
| Risk Types Identified | 4 |
| Data Records Processed | 1.8M+ |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask flask-cors
```

### Step 1: Data Cleaning (Already Done)

Cleaned CSV files are already generated:
- `aadhaar_biometric_cleaned.csv`
- `aadhaar_demographic_cleaned.csv`
- `aadhaar_enrollment_cleaned.csv`

### Step 2: Generate Visualizations

```bash
# Generate all 3 model dashboards
python model1_single_chart.py
python model2_single_chart.py
python model3_single_chart.py
```

**Output:**
- `model1_visuals/MODEL1_COMPREHENSIVE_DASHBOARD.png`
- `model2_visuals/MODEL2_COMPREHENSIVE_DASHBOARD.png`
- `model3_visuals/MODEL3_COMPREHENSIVE_DASHBOARD.png`

### Step 3: Run Interactive Dashboard

```bash
# Terminal 1: Start Flask API
python api.py

# Terminal 2: Open dashboard in browser
start advanced_dashboard.html
```

**Dashboard URL:** `http://localhost:5001`

### Step 4: Train ML Models (Optional)

```bash
python train_model.py
```

**Output:**
- `best_model.pkl`
- `scaler.pkl`
- `model_comparison.json`
- 5 visualization charts

---

## 📝 Key Features

✅ **100% UIDAI Data** - Uses only official Aadhaar datasets  
✅ **3-Model Pipeline** - Anomaly Detection → Risk Scoring → Classification  
✅ **917 Districts** - Comprehensive coverage across India  
✅ **Real-time Dashboard** - Interactive React TypeScript interface  
✅ **Explainable AI** - Clear reasons and recommended actions  
✅ **Production Ready** - Cleaned data, trained models, visualizations  

---

## 📧 Contact

**Team Leader:** Deepak  
**Team Members:** Adarsh Kumar Pandey, Ajay Rajora  
**UIDAI ID:** UIDAI_12208

---

## 📄 License

This project is developed for the UIDAI Hackathon and uses only official UIDAI datasets.

---

**Last Updated:** January 2025  
**Version:** 1.0.0
