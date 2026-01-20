# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import os
import re
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD BIOMETRIC FILES (LOCAL)
# =====================================================
BASE_DIR = "."

files = [
    f for f in os.listdir(BASE_DIR)
    if f.startswith("api_data_aadhar_biometric") and f.endswith(".csv")
]

print("Biometric files found:", files)

dfs = [pd.read_csv(f) for f in files]
bio_raw = pd.concat(dfs, ignore_index=True)

print("Raw biometric rows:", len(bio_raw))
bio_raw.head()

# =====================================================
# 2. STATE CLEANING (SAME STANDARD AS DEMO)
# =====================================================
def normalize_state(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

STATE_MAP = {
    "odisha": "Odisha",
    "orissa": "Odisha",
    "west bengal": "West Bengal",
    "westbengal": "West Bengal",
    "west bangal": "West Bengal",
    "west bengli": "West Bengal",
    "andhra pradesh": "Andhra Pradesh",
    "uttaranchal": "Uttarakhand",
    "jammu and kashmir": "Jammu and Kashmir",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu":
        "Dadra and Nagar Haveli and Daman and Diu",
    "pondicherry": "Puducherry",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "chhatisgarh": "Chhattisgarh",
}

INVALID_STATES = {
    "jaipur","nagpur","balanagar","madanapalle",
    "darbhanga","puttenahalli","raja annamalai puram","100000"
}

bio_raw["_state_norm"] = bio_raw["state"].apply(normalize_state)
bio_raw.loc[bio_raw["_state_norm"].isin(INVALID_STATES), "_state_norm"] = np.nan

bio_raw["state_clean"] = bio_raw["_state_norm"].map(
    lambda x: STATE_MAP.get(x, x.title()) if pd.notna(x) else np.nan
)

bio_raw = bio_raw.dropna(subset=["state_clean"])

VALID_STATES = {
    "Andaman and Nicobar Islands","Andhra Pradesh","Arunachal Pradesh","Assam",
    "Bihar","Chandigarh","Chhattisgarh",
    "Dadra and Nagar Haveli and Daman and Diu","Delhi","Goa","Gujarat",
    "Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka",
    "Kerala","Ladakh","Lakshadweep","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Puducherry","Punjab","Rajasthan",
    "Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand",
    "West Bengal"
}

bio_raw = bio_raw[bio_raw["state_clean"].isin(VALID_STATES)]

print("States after cleaning:", bio_raw["state_clean"].nunique())
assert bio_raw["state_clean"].nunique() == 36

# =====================================================
# 3. DISTRICT CLEANING
# =====================================================
def normalize_text(s):
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", "and")
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.fullmatch(r"\d+", s):
        return np.nan
    return s

GARBAGE_PATTERNS = [
    r"\bnear\b", r"\broad\b", r"\bcolony\b",
    r"\bsector\b", r"\bthana\b", r"\bhospital\b"
]

def is_garbage(s):
    if pd.isna(s) or len(s) <= 2:
        return True
    for p in GARBAGE_PATTERNS:
        if re.search(p, s):
            return True
    return False

DISTRICT_MAP = {
    "ahmed nagar": "ahilyanagar",
    "ahmadnagar": "ahilyanagar",
    "aurangabad": "chhatrapati sambhajinagar",
    "osmanabad": "dharashiv",
    "hugli": "hooghly",
    "haora": "howrah",
    "anugul": "angul",
    "baleswar": "baleshwar",
    "mysore": "mysuru",
    "bellary": "ballari",
    "belgaum": "belagavi",
    "rangareddi": "rangareddy"
}

bio_raw["_district_norm"] = bio_raw["district"].apply(normalize_text)
bio_raw.loc[bio_raw["_district_norm"].apply(is_garbage), "_district_norm"] = np.nan

bio_raw["district_clean"] = bio_raw["_district_norm"].map(
    lambda x: DISTRICT_MAP.get(x, x)
)

bio_raw["district_clean"] = (
    bio_raw["district_clean"]
    .str.title()
    .str.strip()
)

print("Districts after cleaning:", bio_raw["district_clean"].nunique())

# =====================================================
# 4. DATE + FINAL BIOMETRIC DATA
# =====================================================
bio_raw["date"] = pd.to_datetime(bio_raw["date"], errors="coerce")

biometric_clean = bio_raw[
    ["date","pincode","bio_age_5_17","bio_age_17_",
     "state_clean","district_clean"]
].dropna()

biometric_clean = biometric_clean.sort_values(
    ["date","state_clean","district_clean","pincode"]
)

# =====================================================
# 5. SAVE OUTPUT
# =====================================================
biometric_clean.to_csv("aadhaar_biometric_cleaned.csv", index=False)

print("✅ Saved: aadhaar_biometric_cleaned.csv")
print("Final rows:", len(biometric_clean))
print("Final states:", biometric_clean["state_clean"].nunique())
print("Final districts:", biometric_clean["district_clean"].nunique())

# =====================================================
# 6. FEATURE ENGINEERING
# =====================================================
biometric_clean["day"] = biometric_clean["date"].dt.day
biometric_clean["month"] = biometric_clean["date"].dt.month
biometric_clean["year"] = biometric_clean["date"].dt.year
biometric_clean["quarter"] = biometric_clean["date"].dt.quarter
biometric_clean["dayofweek"] = biometric_clean["date"].dt.dayofweek
biometric_clean["dayname"] = biometric_clean["date"].dt.day_name()

biometric_clean["total_reg"] = biometric_clean["bio_age_5_17"] + biometric_clean["bio_age_17_"]

# =====================================================
# 7. EDA
# =====================================================
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Year distribution
year_counts = biometric_clean.groupby("year").size().reset_index(name='count')
sns.barplot(data=year_counts, x='year', y='count', ax=axes[0,0], palette='Blues_d')
axes[0,0].set_title('Biometric Aadhaar Registrations by Year', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Year')
axes[0,0].set_ylabel('Number of Registrations')
axes[0,0].tick_params(axis='x', rotation=45)

# Month distribution
month_counts = biometric_clean.groupby("month").size().reset_index(name='count')
month_counts['month_name'] = month_counts['month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                                                        7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
sns.barplot(data=month_counts, x='count', y='month_name', ax=axes[0,1], palette='Greens_d', orient='h')
axes[0,1].set_title('Biometric Aadhaar Registrations by Month', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Number of Registrations')
axes[0,1].set_ylabel('Month')

# Day of week distribution
day_counts = biometric_clean.groupby("dayname").size().reset_index(name='count')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts['dayname'] = pd.Categorical(day_counts['dayname'], categories=day_order, ordered=True)
day_counts = day_counts.sort_values('dayname')
sns.barplot(data=day_counts, x='count', y='dayname', ax=axes[1,0], palette='Reds_d', orient='h')
axes[1,0].set_title('Biometric Aadhaar Registrations by Day of Week', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Number of Registrations')
axes[1,0].set_ylabel('Day of Week')

# Total registrations over time (simple line plot)
daily_reg = biometric_clean.groupby('date').size().reset_index(name='count')
sns.lineplot(data=daily_reg, x='date', y='count', ax=axes[1,1], color='purple')
axes[1,1].set_title('Daily Biometric Aadhaar Registrations Trend', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Date')
axes[1,1].set_ylabel('Number of Registrations')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_charts_biometric.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("✅ Enhanced biometric charts saved as 'eda_charts_biometric.png'")
