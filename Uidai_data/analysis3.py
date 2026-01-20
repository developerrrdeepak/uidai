# =========================================
# Aadhaar Biometric Data Cleaning Pipeline
# =========================================

import pandas as pd
import re

# -----------------------------------------
# 1. READ & MERGE CSV FILES
# -----------------------------------------

files = [
    "api_data_aadhar_biometric_0_500000.csv",
    "api_data_aadhar_biometric_500000_1000000.csv",
    "api_data_aadhar_biometric_1000000_1500000.csv",
    "api_data_aadhar_biometric_1500000_1861108.csv"
]

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
print("Raw records:", len(df))

# -----------------------------------------
# 2. BASIC STANDARDIZATION
# -----------------------------------------

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
df["new_date"] = df["date"].dt.strftime("%Y%m%d")

# -----------------------------------------
# 3. TEXT CLEANING FUNCTION
# -----------------------------------------

def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).lower()
    x = re.sub(r"[^a-z\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

df["state_clean"] = df["state"].apply(clean_text)
df["district_clean"] = df["district"].apply(clean_text)

# -----------------------------------------
# 4. STATE NORMALIZATION (CRITICAL FIX)
# -----------------------------------------

state_fix = {
    # Wrong values present in data
    "darbhanga": "Bihar",
    "puttenahalli": "Karnataka",
    "balanagar": "Telangana",
    "jaipur": "Rajasthan",
    "madanapalle": "Andhra Pradesh",
    "nagpur": "Maharashtra",
    "raja annamalai puram": "Tamil Nadu",

    # Legacy / spelling issues
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
    "chattisgarh": "Chhattisgarh",
    "telengana": "Telangana",

    # Delhi variants
    "nct of delhi": "Delhi",
    "national capital territory of delhi": "Delhi",

    # J&K
    "jammu kashmir": "Jammu And Kashmir",
    "jammu and kashmir ut": "Jammu And Kashmir",

    # UT merger
    "dadra and nagar haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "daman and diu": "Dadra And Nagar Haveli And Daman And Diu",
    "dadra and nagar haveli and daman and diu":
        "Dadra And Nagar Haveli And Daman And Diu",

    # Others
    "pondicherry": "Puducherry",
    "andaman nicobar islands": "Andaman And Nicobar Islands"
}

df["state_clean"] = df["state_clean"].replace(state_fix)

# -----------------------------------------
# 5. FINAL INDIA STATE FILTER (GUARANTEES 36)
# -----------------------------------------

valid_states = {
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa",
    "Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland",
    "Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal",
    "Delhi","Puducherry","Chandigarh","Ladakh",
    "Jammu And Kashmir","Andaman And Nicobar Islands",
    "Dadra And Nagar Haveli And Daman And Diu","Lakshadweep"
}

df = df[df["state_clean"].isin(valid_states)]
print("Final unique states:", df["state_clean"].nunique())

# -----------------------------------------
# 6. NUMERIC CLEANING
# -----------------------------------------

# Identify biometric columns dynamically
biometric_cols = [col for col in df.columns if any(keyword in col.lower() 
    for keyword in ['bio_', 'fingerprint', 'iris', 'face', 'update'])]

num_cols = ["pincode"] + biometric_cols
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["state_clean", "district_clean", "pincode"])
df = df[df["pincode"].astype(str).str.match(r"^\d{6}$")]

# -----------------------------------------
# 7. DEDUPLICATION
# -----------------------------------------

df = df.drop_duplicates(
    subset=["date", "state_clean", "district_clean", "pincode"]
)

# -----------------------------------------
# 8. FEATURE ENGINEERING
# -----------------------------------------

# Create total biometric updates
if biometric_cols:
    df["total_bio_updates"] = df[biometric_cols].sum(axis=1, skipna=True)
    print(f"Biometric columns found: {biometric_cols}")
else:
    # If no specific biometric columns, create proxy
    df["total_bio_updates"] = 1  # Each row represents one biometric update
    print("No specific biometric columns found, using row count as proxy")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# -----------------------------------------
# 9. FINAL OUTPUT
# -----------------------------------------

print(df.info())
df.to_csv("aadhar_biometric_cleaned_final.csv", index=False)

print("✅ Biometric data cleaning complete")
print("Final shape:", df.shape)