# Financial Inclusion Scout with Aadhaar Early-Warning Intelligence

**A Proactive System for Detecting Aadhaar-Based Financial Inclusion Risks**

---

## 📋 Project Information

**UIDAI ID:** UIDAI_12208
**Team Members:** Deepak (Team Leader), Adarsh Kumar Pandey, Ajay Rajora

---

## 🚀 Quick Start (New Structure)

This project has been restructured for better maintainability. To run the code, use the following commands from the **project root**:

### 1. Main Pipeline
Runs the complete data processing and risk scoring pipeline.
```bash
python -m src.core.main
```

### 2. Risk Assessment Only
```bash
python -m src.models.risk_scoring
```

### 3. Start Dashboard API
```bash
python -m src.api.api
```

### 4. Interactive Dashboard
Open `outputs/dashboards/advanced_dashboard.html` in your browser (after starting the API).

---

## 📂 Project Structure

```
uidai data/
├── src/                       # Source Code
│   ├── core/                  # Main pipeline logic
│   ├── models/                # ML models and risk engines
│   ├── analysis/              # Data analysis and validation
│   ├── api/                   # Backend API
│   └── visualization/         # Plotting scripts
├── data/                      # Data Storage
│   ├── raw/                   # Raw input CSVs
│   └── cleaned/               # Processed data
├── outputs/                   # Generated Outputs
│   ├── dashboards/            # HTML dashboards
│   ├── images/                # Charts and plots
│   └── results/               # JSON/CSV results
├── docs/                      # Documentation
└── scripts/                   # Utility scripts
```

---

## 🎯 Project Overview

### Problem Statement
Aadhaar-based enrollment and update gaps silently block financial inclusion and welfare delivery. Current monitoring detects these risks too late.

### Solution
An explainable AI system that uses Aadhaar data to:
- Identify **high-risk districts**
- Detect **early warning signals**
- Recommend **corrective actions**

### Key Features
1.  **Model 1 (Anomaly Detection):** Detects sudden drops/spikes in enrolment and updates.
2.  **Model 2 (Risk Scoring):** Calculates a composite risk score (0-1) for every district.
3.  **Model 3 (Risk Classification):** Classifies specific risks (e.g., "Child Exclusion", "Gender Gap") and suggests policy actions.

---

## 📊 Methodology

### Pipeline
`Raw Data` → `Data Cleaning` → `Feature Engineering` → `AI Models` → `Actionable Insights`

### Risk Indicators
- **Enrolment Coverage:** Is the district keeping up with state averages?
- **Child Enrolment (0-5):** Are children being excluded?
- **Gender Gap:** Is there equitable access for women?
- **Update Readiness:** Are biometrics being updated to prevent authentication failures?
- **Momentum:** Is enrollment growing or shrinking?

---

## 📈 Results & Impact

- **Districts Analyzed:** 917
- **Anomalies Detected:** ~25% detection rate
- **Impact:** Shift from reactive reporting to preventive decision-making.

---

## 🔧 Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask flask-cors
```
*Note: If you encounter numpy/pandas errors, run `pip install --upgrade pandas numpy`.*

---

**Last Updated:** February 2026
**Version:** 2.0.0 (Restructured)
