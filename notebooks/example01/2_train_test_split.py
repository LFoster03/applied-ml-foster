"""Train-test split for automobile price prediction."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Paths ---
DATA_PATH = Path(__file__).parent / "automobile_price_data3_cleaned.csv"
TRAIN_PATH = Path(__file__).parent / "train_data.csv"
TEST_PATH = Path(__file__).parent / "test_data.csv"

# --- Load cleaned data ---
df = pd.read_csv(DATA_PATH)
print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Define features (X) and target (y) ---
target = "price"
X = df.drop(columns=[target])
y = df[target]

# --- Split into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Data Split Complete")
print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

# --- Combine back into single files for easy inspection ---
train = X_train.copy()
train[target] = y_train
test = X_test.copy()
test[target] = y_test

# --- Save ---
train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)
print(f"\nTraining data saved to: {TRAIN_PATH}")
print(f"Test data saved to: {TEST_PATH}")
