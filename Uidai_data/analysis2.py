# =========================================
# Aadhaar Demographic Data Cleaning Pipeline
# =========================================

import pandas as pd
import re

# -----------------------------------------
# 1. READ & MERGE CSV FILES
# -----------------------------------------

files = [
    "api_data_aadhar_demographic_0_500000.csv",
    "api_data_aadhar_demographic_500000_1000000.csv",
    "api_data_aadhar_demographic_1000000_1500000.csv",
    "api_data_aadhar_demographic_1500000_2000000.csv",
    "api_data_aadhar_demographic_2000000_2071700.csv"
]

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
print("Raw records:", len(df))

# -----------------------------------------
# 2. BASIC STANDARDIZATION
# -----------------------------------------

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.rename(columns={"demo_age_17_": "demo_age_18_plus"})

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

num_cols = ["pincode", "demo_age_5_17", "demo_age_18_plus"]
for c in num_cols:
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

df["total_enroll"] = df["demo_age_5_17"] + df["demo_age_18_plus"]
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# -----------------------------------------
# 9. FINAL OUTPUT
# -----------------------------------------

print(df.info())
df.to_csv("aadhar_demographic_cleaned_final.csv", index=False)

print("✅ Data cleaning complete")
print("Final shape:", df.shape)
