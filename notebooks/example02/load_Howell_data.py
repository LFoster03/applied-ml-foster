"""Step 1: Load and inspect the Howell dataset.

This script:
1. Loads Howell.csv from the same folder as this file.
2. Checks for missing values and duplicates.
3. Prints basic statistics for an initial understanding.
"""

from pathlib import Path
import sys

import pandas as pd

# ✅ Define path to Howell.csv (same folder as this script)
data_path: Path = Path(__file__).parent / "Howell.csv"

# ✅ Check if file exists
if not data_path.exists():
    sys.exit(f"❌ ERROR: Could not find Howell.csv at {data_path}")

# ✅ Load dataset (uses ';' as separator)
howell_df: pd.DataFrame = pd.read_csv(data_path, sep=";")

# ✅ Basic dataset overview
print("✅ Howell dataset loaded successfully!")
print(f"Rows: {howell_df.shape[0]}, Columns: {howell_df.shape[1]}\n")

# ✅ Display the first few rows
print("🔹 First 5 rows:")
print(howell_df.head(), "\n")

# ✅ Check for missing values
print("🔹 Missing values per column:")
print(howell_df.isnull().sum(), "\n")

# ✅ Check for duplicates
duplicates = howell_df.duplicated().sum()
print(f"🔹 Duplicate rows: {duplicates}\n")

# ✅ Summary statistics
print("🔹 Summary statistics:")
print(howell_df.describe(include='all'))

# ✅ Summary statistics
print("🔹 Summary statistics:")
print(howell_df.describe(include='all'))

# ✅ Optional: Save a cleaned version for later
cleaned_path = Path(__file__).parent / "Howell_cleaned.csv"
howell_df.to_csv(cleaned_path, index=False)
print(f"\n💾 Cleaned file saved to: {cleaned_path}")
