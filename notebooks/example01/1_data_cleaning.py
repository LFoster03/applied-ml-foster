"""Data cleaning script for the automobile_price_data3 dataset.

This script loads the CSV file, cleans missing values, converts numeric
columns, and saves a cleaned version for further analysis.
"""

import pandas as pd

# Load the data
df = pd.read_csv(r"C:\Repos\applied-ml-foster\notebooks\example01\automobile_price_data3.csv")

# Preview the first few rows
print(df.head())

# Check structure and missing values
print("\n--- Info ---")
print(df.info())

print("\n--- Missing values ---")
print(df.isna().sum())

# Step 1: Handle missing values
# Replace '?' or empty strings with NaN, then re-check
df.replace("?", pd.NA, inplace=True)
print("\nMissing after replacing '?':")
print(df.isna().sum())

# Step 2: Convert numeric columns to proper types
numeric_cols = ["normalized-losses", "weight", "engine-size", "bhp", "mpg", "price"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Step 3: Drop rows with critical missing values (like price)
df = df.dropna(subset=["price"])

# Step 4: Fill other missing values (e.g., with median for numeric)
df["normalized-losses"].fillna(df["normalized-losses"].median(), inplace=True)

# Step 5: Normalize text (make all lowercase, strip spaces)
df["make"] = df["make"].str.lower().str.strip()
df["fuel"] = df["fuel"].str.lower().str.strip()
df["body"] = df["body"].str.lower().str.strip()
df["drive"] = df["drive"].str.lower().str.strip()

# Step 6: Verify cleaned dataset
print("\n--- Cleaned Data Preview ---")
print(df.head())

# Optional: Save cleaned version
df.to_csv("automobile_price_data3_cleaned.csv", index=False)
print("\n✅ Cleaned file saved as automobile_price_data3_cleaned.csv")
