# 📁 Detailed File Structure & Project Directory

**Project:** Financial Inclusion Scout with Aadhaar Early-Warning Intelligence  
**UIDAI ID:** UIDAI_12208

---

## 📂 Complete Directory Structure

```
uidai data/
│
├── 📄 README.md                                    # Main project documentation
├── 📄 README2.md                                   # This file - Detailed structure
├── 📄 CLEANED_FILES_VERIFICATION.md                # Data verification report
├── 📄 ML_TRAINING_GUIDE.md                         # ML training documentation
│
├── 📊 RAW DATA FILES (Input)
│   ├── api_data_aadhar_enrolment_0_500000.csv      # Enrollment data part 1
│   ├── api_data_aadhar_enrolment_500000_1000000.csv # Enrollment data part 2
│   ├── api_data_aadhar_enrolment_1000000_1006029.csv # Enrollment data part 3
│   ├── api_data_aadhar_demographic_*.csv           # Demographic data files
│   └── api_data_aadhar_biometric_*.csv             # Biometric update data files
│
├── 🧹 DATA CLEANING SCRIPTS
│   ├── analysis.py                                 # Enrollment data cleaning
│   │   ├── Input: api_data_aadhar_enrolment_*.csv
│   │   ├── Output: aadhaar_enrollment_cleaned.csv
│   │   ├── Functions: State name normalization, date parsing
│   │   └── Records: 1,006,029 → Cleaned dataset
│   │
│   ├── analysis2.py                                # Demographic data cleaning
│   │   ├── Input: api_data_aadhar_demographic_*.csv
│   │   ├── Output: aadhaar_demographic_cleaned.csv
│   │   ├── Functions: Age group standardization, gender normalization
│   │   └── Records: 882,744 rows, 814 districts
│   │
│   └── analysis3.py                                # Biometric data cleaning
│       ├── Input: api_data_aadhar_biometric_*.csv
│       ├── Output: aadhaar_biometric_cleaned.csv
│       ├── Functions: Time-series alignment, outlier handling
│       └── Records: 917,008 rows, 907 districts
│
├── 🗂️ CLEANED DATA FILES (Output)
│   ├── aadhaar_enrollment_cleaned.csv              # 1M+ rows, enrollment data
│   │   └── Columns: date, state_clean, district_clean, pincode,
│   │               age_0_5, age_5_17, age_18_greater, total_enrolment
│   │
│   ├── aadhaar_demographic_cleaned.csv             # 882K rows, demographic data
│   │   └── Columns: state, district, age_group, gender, 
│   │               enrolment_count, time_period
│   │
│   └── aadhaar_biometric_cleaned.csv               # 917K rows, biometric updates
│       └── Columns: state, district, time_period, 
│                   biometric_updates, update_type
│
├── 🤖 MACHINE LEARNING MODELS
│   ├── train_model.py                              # ML model training script
│   │   ├── Algorithms: Logistic Regression, Random Forest, 
│   │   │              Gradient Boosting, Decision Tree, SVM, KNN
│   │   ├── Output: best_model.pkl, scaler.pkl
│   │   ├── Metrics: Accuracy, Precision, Recall, F1-Score
│   │   └── Visualizations: 5 comparison charts
│   │
│   ├── complete_unified_system.py                  # Integrated 3-model pipeline
│   │   ├── Model 1: Anomaly Detection (Z-score)
│   │   ├── Model 2: Risk Scoring (Normalized indicators)
│   │   ├── Model 3: Rule-Based Classification
│   │   └── Output: Unified risk assessment
│   │
│   └── rule_based_risk_classification_model3.py    # Model 3 standalone
│       ├── Risk Types: Administrative Disruption, DBT Failure,
│       │              Child Welfare Exclusion, Gender Barrier,
│       │              Migration/Crisis Shock
│       └── Output: Risk type + Recommended action
│
├── 📈 VISUALIZATION GENERATORS
│   ├── model1_single_chart.py                      # Model 1 dashboard generator
│   │   ├── Input: aadhaar_biometric_cleaned.csv, 
│   │   │         aadhaar_demographic_cleaned.csv
│   │   ├── Output: MODEL1_COMPREHENSIVE_DASHBOARD.png
│   │   ├── Charts: KPI cards, coverage histogram, scatter plot,
│   │   │          top anomalies, statistics table
│   │   └── Size: 1920x1080px, 300 DPI
│   │
│   ├── model2_single_chart.py                      # Model 2 dashboard generator
│   │   ├── Input: Cleaned datasets + risk scores
│   │   ├── Output: MODEL2_COMPREHENSIVE_DASHBOARD.png
│   │   ├── Charts: Risk level KPIs, distribution, donut chart,
│   │   │          top high-risk districts, statistics
│   │   └── Size: 1920x1080px, 300 DPI
│   │
│   └── model3_single_chart.py                      # Model 3 dashboard generator
│       ├── Input: Cleaned datasets + classifications
│       ├── Output: MODEL3_COMPREHENSIVE_DASHBOARD.png
│       ├── Charts: Severity KPIs, risk type distribution,
│       │          stacked bars, top districts, rule logic
│       └── Size: 1920x1080px, 300 DPI
│
├── 🖼️ GENERATED VISUALIZATIONS
│   ├── model1_visuals/
│   │   ├── MODEL1_COMPREHENSIVE_DASHBOARD.png      # Main Model 1 dashboard
│   │   ├── anomaly_detection_scatter.png           # Scatter plot
│   │   ├── coverage_distribution.png               # Histogram
│   │   └── top_anomalies.png                       # Bar chart
│   │
│   ├── model2_visuals/
│   │   ├── MODEL2_COMPREHENSIVE_DASHBOARD.png      # Main Model 2 dashboard
│   │   ├── risk_score_distribution.png             # Distribution plot
│   │   ├── risk_level_donut.png                    # Donut chart
│   │   └── high_risk_districts.png                 # Bar chart
│   │
│   └── model3_visuals/
│       ├── MODEL3_COMPREHENSIVE_DASHBOARD.png      # Main Model 3 dashboard
│       ├── risk_type_distribution.png              # Bar chart
│       ├── severity_stacked.png                    # Stacked bar
│       └── classification_results.png              # Results table
│
├── 🌐 DASHBOARD & API
│   ├── api.py                                      # Flask REST API
│   │   ├── Port: 5001
│   │   ├── Endpoints: /api/districts, /api/stats, /api/high-risk
│   │   ├── CORS: Enabled
│   │   └── Data: Serves cleaned datasets as JSON
│   │
│   ├── advanced_dashboard.html                     # React TypeScript dashboard
│   │   ├── Framework: React 18 + Babel
│   │   ├── Features: 4 tabs, search, filter, export, pagination
│   │   ├── Theme: Dark mode with glassmorphism
│   │   ├── Charts: Chart.js integration
│   │   └── API: Connects to Flask backend (port 5001)
│   │
│   └── app.py                                      # Streamlit dashboard (alternative)
│       ├── Framework: Streamlit
│       ├── Features: Multi-page app, interactive filters
│       └── Port: 8501 (default)
│
├── 🖥️ SCREENSHOTS
│   ├── react_dashboard.png                         # React dashboard screenshot
│   ├── streamlit_overview.png                      # Streamlit overview
│   └── api_response.png                            # API response example
│
├── 💾 TRAINED MODEL FILES
│   ├── best_model.pkl                              # Best performing ML model
│   ├── scaler.pkl                                  # Feature scaler
│   ├── model_comparison.json                       # Model performance metrics
│   └── feature_importance.csv                      # Feature importance scores
│
└── 📋 CONFIGURATION FILES
    ├── requirements.txt                            # Python dependencies
    ├── .gitignore                                  # Git ignore rules
    └── config.json                                 # Application configuration
```

---

## 📊 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW UIDAI DATA FILES                     │
│  (Enrollment, Demographic, Biometric - 1.8M+ records)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA CLEANING SCRIPTS                          │
│  analysis.py → analysis2.py → analysis3.py                  │
│  • State normalization  • Missing value handling            │
│  • Date parsing         • Outlier detection                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              CLEANED DATA FILES (CSV)                       │
│  • aadhaar_enrollment_cleaned.csv                           │
│  • aadhaar_demographic_cleaned.csv                          │
│  • aadhaar_biometric_cleaned.csv                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              3-MODEL PIPELINE                               │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Model 1: Anomaly Detection (Z-score)            │        │
│  │ Output: 229 anomalies detected                  │        │
│  └─────────────────────────────────────────────────┘        │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Model 2: Risk Scoring (Normalized)              │        │
│  │ Output: 73 High, 39 Medium, 74 Low Risk         │        │
│  └─────────────────────────────────────────────────┘        │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Model 3: Rule-Based Classification              │        │
│  │ Output: Risk type + Recommended action          │        │
│  └─────────────────────────────────────────────────┘        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              VISUALIZATION LAYER                            │
│  • model1_single_chart.py → MODEL1_DASHBOARD.png            │
│  • model2_single_chart.py → MODEL2_DASHBOARD.png            │
│  • model3_single_chart.py → MODEL3_DASHBOARD.png            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              DASHBOARD & API LAYER                          │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   Flask API      │◄────────┤  React Dashboard │          │
│  │   (port 5001)    │         │  (HTML/JS/CSS)   │          │
│  └──────────────────┘         └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Script Dependencies

### analysis.py
```
Input Files:
  - api_data_aadhar_enrolment_0_500000.csv
  - api_data_aadhar_enrolment_500000_1000000.csv
  - api_data_aadhar_enrolment_1000000_1006029.csv

Output Files:
  - aadhaar_enrollment_cleaned.csv

Dependencies:
  - pandas
  - matplotlib
  - numpy

Key Functions:
  - State name normalization
  - Date parsing and validation
  - Total enrollment calculation
  - Top/Bottom state ranking
  - Time-series visualization
```

### analysis2.py
```
Input Files:
  - api_data_aadhar_demographic_*.csv

Output Files:
  - aadhaar_demographic_cleaned.csv

Dependencies:
  - pandas
  - numpy

Key Functions:
  - Age group standardization
  - Gender normalization
  - District-level aggregation
  - Child enrollment ratio calculation
```

### analysis3.py
```
Input Files:
  - api_data_aadhar_biometric_*.csv

Output Files:
  - aadhaar_biometric_cleaned.csv

Dependencies:
  - pandas
  - numpy

Key Functions:
  - Time-series alignment
  - Biometric update trend analysis
  - Rolling window imputation
  - Update readiness calculation
```

### train_model.py
```
Input Files:
  - aadhaar_enrollment_cleaned.csv
  - aadhaar_demographic_cleaned.csv
  - aadhaar_biometric_cleaned.csv

Output Files:
  - best_model.pkl
  - scaler.pkl
  - model_comparison.json
  - 5 visualization charts

Dependencies:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

Algorithms:
  1. Logistic Regression
  2. Random Forest Classifier
  3. Gradient Boosting Classifier
  4. Decision Tree Classifier
  5. Support Vector Machine (SVM)
  6. K-Nearest Neighbors (KNN)
```

### complete_unified_system.py
```
Input Files:
  - aadhaar_enrollment_cleaned.csv
  - aadhaar_demographic_cleaned.csv
  - aadhaar_biometric_cleaned.csv

Output:
  - Console output with risk assessment
  - Integrated 3-model results

Dependencies:
  - pandas
  - numpy
  - scikit-learn

Models:
  - Model 1: Anomaly Detection
  - Model 2: Risk Scoring
  - Model 3: Rule-Based Classification
```

### model1_single_chart.py
```
Input Files:
  - aadhaar_biometric_cleaned.csv
  - aadhaar_demographic_cleaned.csv

Output Files:
  - model1_visuals/MODEL1_COMPREHENSIVE_DASHBOARD.png

Dependencies:
  - pandas
  - matplotlib
  - seaborn
  - numpy

Chart Components:
  - 4 KPI cards (Total Districts, Anomalies, Normal, Rate)
  - Coverage distribution histogram
  - Biometric vs Demographic scatter plot
  - Top 10 anomalous districts bar chart
  - Detailed statistics table
```

### model2_single_chart.py
```
Input Files:
  - aadhaar_enrollment_cleaned.csv
  - aadhaar_demographic_cleaned.csv
  - aadhaar_biometric_cleaned.csv

Output Files:
  - model2_visuals/MODEL2_COMPREHENSIVE_DASHBOARD.png

Dependencies:
  - pandas
  - matplotlib
  - seaborn
  - numpy

Chart Components:
  - 3 Risk level KPI cards (High/Medium/Low)
  - Risk score distribution by level
  - Risk level proportion donut chart
  - Top 12 high-risk districts (color-coded)
  - Risk scoring statistics table
```

### model3_single_chart.py
```
Input Files:
  - aadhaar_enrollment_cleaned.csv
  - aadhaar_demographic_cleaned.csv
  - aadhaar_biometric_cleaned.csv

Output Files:
  - model3_visuals/MODEL3_COMPREHENSIVE_DASHBOARD.png

Dependencies:
  - pandas
  - matplotlib
  - seaborn
  - numpy

Chart Components:
  - 3 Severity KPI cards (High/Medium/Low)
  - Risk type distribution bar chart
  - Severity by risk type stacked bar
  - Top 10 high-risk districts (color = risk type)
  - Classification statistics and rule logic
```

### api.py
```
Input Files:
  - aadhaar_enrollment_cleaned.csv
  - aadhaar_demographic_cleaned.csv
  - aadhaar_biometric_cleaned.csv

Dependencies:
  - flask
  - flask-cors
  - pandas

Endpoints:
  GET /api/districts       - All district data
  GET /api/stats          - Summary statistics
  GET /api/high-risk      - High-risk districts only
  GET /api/search?q=name  - Search districts

Port: 5001
CORS: Enabled for localhost
```

### advanced_dashboard.html
```
Dependencies:
  - React 18 (CDN)
  - Babel Standalone (CDN)
  - Chart.js (CDN)
  - Fetch API

Features:
  - 4 Interactive tabs
  - Real-time search
  - Risk level filtering
  - Export to CSV
  - Pagination (20 items/page)
  - Dark theme with glassmorphism

API Connection:
  - Backend: http://localhost:5001
  - Auto-refresh on data change
```

---

## 📦 File Size Information

| File | Size | Records | Districts |
|------|------|---------|-----------|
| aadhaar_enrollment_cleaned.csv | ~150 MB | 1,006,029 | 750+ |
| aadhaar_demographic_cleaned.csv | ~80 MB | 882,744 | 814 |
| aadhaar_biometric_cleaned.csv | ~90 MB | 917,008 | 907 |
| best_model.pkl | ~5 MB | - | - |
| MODEL1_DASHBOARD.png | ~2 MB | - | - |
| MODEL2_DASHBOARD.png | ~2 MB | - | - |
| MODEL3_DASHBOARD.png | ~2 MB | - | - |

---

## 🚀 Execution Order

### Complete Pipeline Execution

```bash
# Step 1: Data Cleaning (Run in order)
python analysis.py      # Creates aadhaar_enrollment_cleaned.csv
python analysis2.py     # Creates aadhaar_demographic_cleaned.csv
python analysis3.py     # Creates aadhaar_biometric_cleaned.csv

# Step 2: Model Training (Optional)
python train_model.py   # Creates best_model.pkl, scaler.pkl

# Step 3: Generate Visualizations
python model1_single_chart.py  # Creates MODEL1_DASHBOARD.png
python model2_single_chart.py  # Creates MODEL2_DASHBOARD.png
python model3_single_chart.py  # Creates MODEL3_DASHBOARD.png

# Step 4: Run Dashboard
python api.py           # Start Flask API on port 5001
# Then open advanced_dashboard.html in browser
```

### Quick Start (Pre-cleaned Data)

```bash
# If cleaned CSV files already exist:

# Generate visualizations
python model1_single_chart.py
python model2_single_chart.py
python model3_single_chart.py

# Start dashboard
python api.py
start advanced_dashboard.html
```

---

## 🔍 Key Metrics by File

### aadhaar_enrollment_cleaned.csv
- Total Records: 1,006,029
- States/UTs: 36
- Districts: 750+
- Date Range: 2015-2024
- Total Enrollments: 1.3+ Billion

### aadhaar_demographic_cleaned.csv
- Total Records: 882,744
- Districts: 814
- Age Groups: 3 (0-5, 5-17, 18+)
- Gender Categories: 2 (Male, Female)

### aadhaar_biometric_cleaned.csv
- Total Records: 917,008
- Districts: 907
- Update Types: Multiple
- Time Period: Monthly aggregation

### Model Results
- Districts Analyzed: 917
- Anomalies Detected: 229 (25%)
- High Risk Districts: 73
- Medium Risk Districts: 39
- Low Risk Districts: 74
- Risk Types: 4

---

## 📝 File Naming Conventions

### Data Files
- `aadhaar_*_cleaned.csv` - Cleaned datasets
- `api_data_aadhar_*_*.csv` - Raw UIDAI data

### Model Files
- `*_model.pkl` - Trained ML models
- `*_scaler.pkl` - Feature scalers
- `model_comparison.json` - Performance metrics

### Visualization Files
- `MODEL*_COMPREHENSIVE_DASHBOARD.png` - Main dashboards
- `*_distribution.png` - Distribution charts
- `*_scatter.png` - Scatter plots
- `*_bar.png` - Bar charts

### Script Files
- `analysis*.py` - Data cleaning scripts
- `train_*.py` - Model training scripts
- `model*_single_chart.py` - Visualization generators
- `*_model*.py` - Model implementation scripts

---

## 🛠️ Technology Stack by Component

### Data Processing
- **Language:** Python 3.11
- **Libraries:** Pandas, NumPy
- **Format:** CSV

### Machine Learning
- **Framework:** Scikit-learn
- **Algorithms:** 6 classifiers
- **Serialization:** Pickle

### Visualization
- **Static:** Matplotlib, Seaborn
- **Interactive:** Chart.js
- **Format:** PNG (300 DPI)

### Dashboard
- **Frontend:** React 18 + TypeScript (Babel)
- **Backend:** Flask + Flask-CORS
- **Alternative:** Streamlit

### API
- **Framework:** Flask
- **Protocol:** REST
- **Format:** JSON
- **Port:** 5001

---

## 📧 Contact & Support

**Team Leader:** Deepak  
**Team Members:** Adarsh Kumar Pandey, Ajay Rajora  
**UIDAI ID:** UIDAI_12208

---

**Last Updated:** January 2025  
**Version:** 1.0.0
