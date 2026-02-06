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
# 1. LOAD ALL DEMOGRAPHIC FILES
# =====================================================
BASE_DIR = "."

files = [
    f for f in os.listdir(BASE_DIR)
    if f.startswith("api_data_aadhar_demographic") and f.endswith(".csv")
]

print("Files detected:", len(files))

dfs = [pd.read_csv(f) for f in files]
df_raw = pd.concat(dfs, ignore_index=True)

print("Raw rows:", len(df_raw))

# =====================================================
# 2. STATE CLEANING (EXACTLY 36)
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
    "uttaranchal": "Uttarakhand",
    "pondicherry": "Puducherry",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu":
        "Dadra and Nagar Haveli and Daman and Diu",
    "jammu and kashmir": "Jammu and Kashmir",
    "chhatisgarh": "Chhattisgarh",
}

INVALID_STATES = {
    "jaipur","nagpur","balanagar","madanapalle",
    "darbhanga","puttenahalli","raja annamalai puram","100000"
}

df_raw["_state_norm"] = df_raw["state"].apply(normalize_state)
df_raw.loc[df_raw["_state_norm"].isin(INVALID_STATES), "_state_norm"] = np.nan

df_raw["state_clean"] = df_raw["_state_norm"].map(
    lambda x: STATE_MAP.get(x, x.title()) if pd.notna(x) else np.nan
)

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

df_raw = df_raw[df_raw["state_clean"].isin(VALID_STATES)]
assert df_raw["state_clean"].nunique() == 36

print("States after cleaning:", df_raw["state_clean"].nunique())

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

DIRECTIONAL = {
    "east","west","north","south",
    "north east","north west","south east","south west"
}

def is_garbage(s):
    if pd.isna(s) or len(s) <= 2:
        return True
    if s in DIRECTIONAL:
        return True
    return any(re.search(p, s) for p in GARBAGE_PATTERNS)

DISTRICT_MAP = {
    "ahmed nagar": "ahilyanagar",
    "ahmadnagar": "ahilyanagar",
    "aurangabad": "chhatrapati sambhajinagar",
    "osmanabad": "dharashiv",
    "bid": "beed",
    "mysore": "mysuru",
    "bellary": "ballari",
    "belgaum": "belagavi",
    "shimoga": "shivamogga",
    "chickmagalur": "chikkamagaluru",
    "chikmagalur": "chikkamagaluru",
    "anugul": "angul",
    "baleswar": "baleshwar",
    "hugli": "hooghly",
    "haora": "howrah",
    "tuticorin": "thoothukkudi",
    "tiruvallur": "thiruvallur",
    "tiruvarur": "thiruvarur",
    "villupuram": "viluppuram",
    "bara banki": "barabanki",
    "bulandshahar": "bulandshahr",
    "jyotiba phule nagar": "amroha",
    "sant ravidas nagar bhadohi": "bhadohi",
    "rangareddi": "rangareddy",
    "ysr": "ysr kadapa",
    "cuddapah": "ysr kadapa",
    "east delhi": "delhi",
    "west delhi": "delhi",
    "north delhi": "delhi",
    "south delhi": "delhi",
    "north east delhi": "delhi",
    "north west delhi": "delhi",
    "south east delhi": "delhi",
    "south west delhi": "delhi",
    "mumbai city": "mumbai",
    "mumbai suburban": "mumbai",
    "bangalore urban": "bengaluru",
    "bangalore rural": "bengaluru",
    "purnea": "purnia",
    "rae bareli": "raebareli",
}

df_raw["_district_norm"] = df_raw["district"].apply(normalize_text)
df_raw.loc[df_raw["_district_norm"].apply(is_garbage), "_district_norm"] = np.nan

df_raw["district_clean"] = df_raw["_district_norm"].map(
    lambda x: DISTRICT_MAP.get(x, x)
)

df_raw["district_clean"] = df_raw["district_clean"].str.title().str.strip()

# =====================================================
# 3A. ADVANCED DISTRICT MERGING (STATE-AWARE)
# =====================================================
def merge_district_variants(row):
    """
    Merge district spelling variants while keeping state-specific districts separate.
    Returns canonical district name based on state context.
    """
    state = row["state_clean"]
    district = row["district_clean"]
    
    if pd.isna(district):
        return district
    
    district_lower = district.lower()
    
    # ========== ANDHRA PRADESH ==========
    if state == "Andhra Pradesh":
        if district_lower in ["ananthapur", "ananthapuramu"]:
            return "Anantapur"
        if district_lower in ["y s r", "cuddapah", "ysr"]:
            return "Ysr Kadapa"
        if district_lower in ["mahabubnagar", "mahbubnagar"]:
            return "Mahabubnagar"
        if district_lower in ["karimnagar", "karim nagar"]:
            return "Karimnagar"
        if district_lower in ["k v rangareddy", "rangareddy", "ranga reddy"]:
            return "Rangareddy"
    
    # ========== BIHAR ==========
    if state == "Bihar":
        if district_lower in ["samstipur", "samastipur"]:
            return "Samastipur"
        if district_lower in ["sheikpura", "sheikhpura"]:
            return "Sheikhpura"
        if district_lower in ["bhabua", "kaimur"]:
            return "Kaimur"
        if district_lower in ["monghyr", "munger"]:
            return "Munger"
        # Note: Chhatrapati Sambhajinagar in Bihar is different from Maharashtra
        if district_lower in ["chhatrapati sambhajinagar", "aurangabad"]:
            return "Aurangabad"  # Keep as Aurangabad for Bihar
    
    # ========== CHHATTISGARH ==========
    if state == "Chhattisgarh":
        if district_lower in ["janjgir champa", "janjgir-champa", "janjgirchampa"]:
            return "Janjgir-Champa"
        if district_lower in ["mohla manpur", "mohla-manpur", "mohlamanpur", 
                              "mohla manpur ambagarh chowki"]:
            return "Mohla-Manpur"
    
    # ========== GUJARAT ==========
    if state == "Gujarat":
        if district_lower in ["ahmadabad", "ahmedabad"]:
            return "Ahmedabad"
        if district_lower in ["dohad", "dahod"]:
            return "Dahod"
        if district_lower in ["banaskantha", "banas kantha"]:
            return "Banaskantha"
        if district_lower in ["panchmahals", "panch mahals"]:
            return "Panchmahal"
        if district_lower in ["sabarkantha", "sabar kantha"]:
            return "Sabarkantha"
    
    # ========== HIMACHAL PRADESH ==========
    if state == "Himachal Pradesh":
        if district_lower in ["lahaul and spiti", "lahul and spiti", "lahaul spiti"]:
            return "Lahaul And Spiti"
    
    # ========== JHARKHAND ==========
    if state == "Jharkhand":
        if district_lower in ["hazaribag", "hazaribagh"]:
            return "Hazaribagh"
        if district_lower in ["pakur", "pakaur"]:
            return "Pakur"
    
    # ========== JAMMU AND KASHMIR ==========
    if state == "Jammu and Kashmir":
        if district_lower in ["badgam", "budgam"]:
            return "Budgam"
        if district_lower in ["baramula", "baramulla"]:
            return "Baramulla"
        if district_lower in ["leh", "leh ladakh"]:
            return "Leh"
    
    # ========== KARNATAKA ==========
    if state == "Karnataka":
        if district_lower in ["chamrajanagar", "chamrajnagar", "chamarajanagar"]:
            return "Chamarajanagar"
        if district_lower in ["hasan", "hassan"]:
            return "Hassan"
        if district_lower in ["tumkur", "tumakuru"]:
            return "Tumakuru"
        if district_lower in ["davangere", "davanagere"]:
            return "Davanagere"
    
    # ========== MAHARASHTRA ==========
    if state == "Maharashtra":
        if district_lower in ["ahmednagar", "ahmed nagar", "ahilyanagar"]:
            return "Ahilyanagar"
        if district_lower in ["gondiya", "gondia"]:
            return "Gondia"
        if district_lower in ["mumbai sub urban", "mumbai suburban"]:
            return "Mumbai"
        if district_lower in ["buldhana", "buldana"]:
            return "Buldhana"
        if district_lower in ["chhatrapati sambhajinagar", "aurangabad", "sambhajinagar"]:
            return "Chhatrapati Sambhajinagar"
        if district_lower in ["raigarh", "raigarh mh", "raigad"]:
            return "Raigarh"
    
    # ========== ODISHA ==========
    if state == "Odisha":
        if district_lower in ["anugal", "angul"]:
            return "Angul"
        if district_lower in ["khorda", "khordha", "khurda"]:
            return "Khordha"
        if district_lower in ["balasore", "baleshwar", "baleswar"]:
            return "Baleshwar"
        if district_lower in ["jajpur", "jajapur"]:
            return "Jajpur"
        if district_lower in ["sundergarh", "sundargarh"]:
            return "Sundargarh"
    
    # ========== PUNJAB ==========
    if state == "Punjab":
        if district_lower in ["firozpur", "ferozepur"]:
            return "Ferozepur"
        if district_lower in ["sas nagar mohali", "s a s nagar mohali", "sahibzada ajit singh nagar"]:
            return "S.A.S. Nagar"
    
    # ========== RAJASTHAN ==========
    if state == "Rajasthan":
        if district_lower in ["jhunjhunu", "jhunjhunun"]:
            return "Jhunjhunu"
        if district_lower in ["jalor", "jalore"]:
            return "Jalore"
        if district_lower in ["dholpur", "dhaulpur"]:
            return "Dholpur"
    
    # ========== TAMIL NADU ==========
    if state == "Tamil Nadu":
        if district_lower in ["kanniyakumari", "kanyakumari"]:
            return "Kanyakumari"
        if district_lower in ["tirupathur", "tirupattur"]:
            return "Tirupattur"
        if district_lower in ["kanchipuram", "kancheepuram"]:
            return "Kanchipuram"
    
    # ========== TELANGANA ==========
    if state == "Telangana":
        if district_lower in ["warangal rural", "warangal urban"]:
            return "Warangal"
        if district_lower in ["medchal malkajgiri", "medchal-malkajgiri", "malkajgiri"]:
            return "Medchal-Malkajgiri"
        if district_lower in ["jangaon", "jangoan"]:
            return "Jangaon"
        if district_lower in ["mahabubnagar", "mahbubnagar"]:
            return "Mahabubnagar"
        if district_lower in ["karimnagar", "karim nagar"]:
            return "Karimnagar"
        if district_lower in ["k v rangareddy", "rangareddy", "ranga reddy"]:
            return "Rangareddy"
    
    # ========== UTTAR PRADESH ==========
    if state == "Uttar Pradesh":
        if district_lower in ["baghpat", "bagpat"]:
            return "Baghpat"
        if district_lower in ["maharajganj", "mahr ajganj"]:
            return "Maharajganj"
    
    # ========== WEST BENGAL ==========
    if state == "West Bengal":
        if district_lower in ["barddhaman", "bardhaman", "burdwan"]:
            return "Bardhaman"
        if district_lower in ["darjiling", "darjeeling"]:
            return "Darjeeling"
        if district_lower in ["puruliya", "purulia"]:
            return "Purulia"
        if district_lower in ["purba bardhaman", "paschim bardhaman"]:
            return "Bardhaman"
        if district_lower in ["purba medinipur", "paschim medinipur", "west medinipur", 
                              "east midnapore", "east midnapur", "medinipur"]:
            return "Medinipur"
        if district_lower in ["hugli", "hooghly"]:
            return "Hooghly"
        if district_lower in ["24 parganas", "north 24 parganas", "north twenty four parganas"]:
            return "North 24 Parganas"
        if district_lower in ["south 24 parganas", "south twenty four parganas"]:
            return "South 24 Parganas"
        if district_lower in ["malda", "maldah"]:
            return "Malda"
    
    return district

df_raw["district_clean"] = df_raw.apply(merge_district_variants, axis=1)

# =====================================================
# 3B. PRINT ALL DISTRICTS (AFTER MERGING, BEFORE FILTER)
# =====================================================
print("\n=== ALL DISTRICT NAMES (AFTER MERGING, BEFORE FREQUENCY FILTER) ===")

all_districts = sorted(
    df_raw["district_clean"]
    .dropna()
    .unique()
)

for d in all_districts:
    print(d)

print("\nTotal districts AFTER merging:", len(all_districts))

# =====================================================
# 3C. FREQUENCY FILTER (ML SAFE)
# =====================================================
district_counts = df_raw["district_clean"].value_counts()
valid_districts = district_counts[district_counts >= 5].index
df_raw = df_raw[df_raw["district_clean"].isin(valid_districts)]

print("Final districts AFTER filter:", df_raw["district_clean"].nunique())

# =====================================================
# 4. DATE + FINAL DATASET
# =====================================================
df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

df = df_raw[
    ["date","pincode","demo_age_5_17","demo_age_17_",
     "state_clean","district_clean"]
].dropna()

df = df.sort_values(
    ["state_clean","district_clean","pincode","date"]
)

# =====================================================
# 5. FEATURE ENGINEERING
# =====================================================
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayname"] = df["date"].dt.day_name()
df["quarter"] = df["date"].dt.quarter

df["total_reg"] = df["demo_age_5_17"] + df["demo_age_17_"]

# =====================================================
# 6. CREATE UNIQUE DISTRICT KEY
# =====================================================
df["unique_district"] = df["state_clean"] + " - " + df["district_clean"]

# =====================================================
# 7. SAVE CLEAN DATA
# =====================================================
df.to_csv("aadhaar_demographic_cleaned.csv", index=False)

print("\n✅ Saved: aadhaar_demographic_cleaned.csv")
print("Final rows:", len(df))
print("Final states:", df["state_clean"].nunique())
print("Final districts:", df["district_clean"].nunique())
print("Final unique districts (state+district):", df["unique_district"].nunique())

# =====================================================
# 8. CREATE STATE-WISE DISTRICT LIST
# =====================================================
print("\n=== GENERATING STATE-WISE DISTRICT LIST ===")

# Create a comprehensive state-district mapping
state_district_map = df.groupby("state_clean")["district_clean"].apply(
    lambda x: sorted(x.unique())
).to_dict()

# Create DataFrame for export
state_district_list = []
for state in sorted(state_district_map.keys()):
    districts = state_district_map[state]
    for district in districts:
        state_district_list.append({
            "State": state,
            "District": district,
            "Total_Registrations": len(df[(df["state_clean"] == state) & 
                                          (df["district_clean"] == district)])
        })

df_state_districts = pd.DataFrame(state_district_list)

# Save to CSV
df_state_districts.to_csv("state_wise_district_list.csv", index=False)
print("✅ Saved: state_wise_district_list.csv")

# Also create a formatted text file
with open("state_wise_district_list.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("STATE-WISE DISTRICT LIST\n")
    f.write("Aadhaar Demographic Data Analysis\n")
    f.write("=" * 80 + "\n\n")
    
    for state in sorted(state_district_map.keys()):
        districts = state_district_map[state]
        f.write(f"\n{state} ({len(districts)} districts)\n")
        f.write("-" * 80 + "\n")
        for i, district in enumerate(districts, 1):
            reg_count = len(df[(df["state_clean"] == state) & 
                              (df["district_clean"] == district)])
            f.write(f"  {i:2d}. {district:<40} ({reg_count:,} registrations)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"Total States: {len(state_district_map)}\n")
    f.write(f"Total Districts: {sum(len(v) for v in state_district_map.values())}\n")
    f.write("=" * 80 + "\n")

print("✅ Saved: state_wise_district_list.txt")

# Print summary to console
print("\n=== STATE-WISE DISTRICT COUNT SUMMARY ===")
for state in sorted(state_district_map.keys()):
    print(f"{state}: {len(state_district_map[state])} districts")

# =====================================================
# 9. DISTRICT SUMMARY BY STATE
# =====================================================
print("\n=== DISTRICT COUNT BY STATE ===")
district_by_state = df.groupby("state_clean")["district_clean"].nunique().sort_values(ascending=False)
for state, count in district_by_state.items():
    print(f"{state}: {count} districts")

# =====================================================
# 10. QUICK EDA
# =====================================================
sns.set_style("whitegrid")

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Plot 1: Registrations by Year
plt.figure(figsize=(10,5))
df.groupby("year").size().plot(kind="bar", color="steelblue")
plt.title("Aadhaar Registrations by Year", fontsize=14, fontweight="bold")
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Registrations", fontsize=12)
plt.tight_layout()
plt.savefig("images/registrations_by_year.jpg", dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: Top 20 Districts by Registration
plt.figure(figsize=(12,6))
top_districts = df["unique_district"].value_counts().head(20)
top_districts.plot(kind="barh", color="coral")
plt.title("Top 20 Districts by Aadhaar Registrations", fontsize=14, fontweight="bold")
plt.xlabel("Number of Registrations", fontsize=12)
plt.ylabel("District (State - District)", fontsize=12)
plt.tight_layout()
plt.savefig("images/top_districts.jpg", dpi=300, bbox_inches="tight")
plt.close()

# Plot 3: Districts per State
plt.figure(figsize=(14,8))
district_by_state.plot(kind="barh", color="mediumseagreen")
plt.title("Number of Districts per State", fontsize=14, fontweight="bold")
plt.xlabel("Number of Districts", fontsize=12)
plt.ylabel("State", fontsize=12)
plt.tight_layout()
plt.savefig("images/districts_per_state.jpg", dpi=300, bbox_inches="tight")
plt.close()

print("\n📊 EDA plots saved:")
print("  - images/registrations_by_year.jpg")
print("  - images/top_districts.jpg")
print("  - images/districts_per_state.jpg")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. aadhaar_demographic_cleaned.csv - Main cleaned dataset")
print("  2. state_wise_district_list.csv - State-wise district list (CSV format)")
print("  3. state_wise_district_list.txt - State-wise district list (formatted text)")
print("  4. images/registrations_by_year.jpg - Visualization")
print("  5. images/top_districts.jpg - Visualization")
print("  6. images/districts_per_state.jpg - Visualization")
print("=" * 80)