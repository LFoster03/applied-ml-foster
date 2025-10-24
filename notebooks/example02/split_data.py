# Prepare split

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load cleaned CSV
data_path = Path(r"C:\Repos\applied-ml-foster\notebooks\example02\Howell_cleaned.csv")
howell_df = pd.read_csv(data_path, sep=",")  # CSV is comma-separated

# 2️⃣ Calculate BMI (if not already present)
howell_df["BMI"] = howell_df["weight"] / (howell_df["height"] / 100) ** 2


# 3️⃣ Add BMI category
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


howell_df["bmi_category"] = howell_df["BMI"].apply(bmi_category)

# 4️⃣ Filter adults only (age >= 18)
howell_adult = howell_df[howell_df["age"] >= 18]

# 5️⃣ Features and target for modeling
X = howell_adult[["height", "weight", "age"]]
y = howell_adult["bmi_category"]

# 6️⃣ Keep only categories with at least 2 samples to avoid stratification errors
valid_categories = y.value_counts()[lambda x: x >= 2].index
mask = y.isin(valid_categories)
X = X[mask]
y = y[mask]

# 7️⃣ Stratified train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8️⃣ Combine features and target
train_data = X_train.copy()
train_data["bmi_category"] = y_train

test_data = X_test.copy()
test_data["bmi_category"] = y_test

# 9️⃣ Save train/test CSVs
train_data.to_csv(r"C:\Repos\applied-ml-foster\notebooks\example02\train_data.csv", index=False)
test_data.to_csv(r"C:\Repos\applied-ml-foster\notebooks\example02\test_data.csv", index=False)

# 1️⃣0️⃣ Sanity check
print("Training set shape:", train_data.shape)
print("Test set shape:", test_data.shape)
print("\nTraining distribution:\n", y_train.value_counts())
print("\nTest distribution:\n", y_test.value_counts())
