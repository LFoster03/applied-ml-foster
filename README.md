# applied-ml-foster

> Start a professional Python project.
>
> Clone repo to local
> Install Extensions
> Set up virtual environment
> Git Commit
> Git pull origin main
> Run Tests: 
uv sync --extra dev --extra docs --upgrade
uv cache clean
git add .
uvx ruff check --fix
uvx pre-commit autoupdate
uv run pre-commit run --all-files
git add .
uv run pytest

# Step 1 for Example01: Data Cleaning Workflow

The cleaning process follows these steps:

1. Load the Dataset, Identify Missing Values, Convert Numeric Columns, Remove Rows Missing the Target Variable, Fill Missing Numeric Values, Fill Missing Categorical Values, Standardize Text Columns, Convert Text Numbers to Numeric, Save the Cleaned Data

- Cleaned data saved to: automobile_price_data3_cleaned.csv

# Step 2: Train–Test Split

Once the data is cleaned, the next step is to **prepare it for machine learning** by splitting it into a training set and a test set.

---

## How It Works

The dataset is divided into:
- **Training set (80%)** — used to train the model  
- **Test set (20%)** — held out for final evaluation  

This ensures that model performance is measured on unseen data.

---

## Script Overview: `2_train_test_split.py`

### 1. **Load Cleaned Data**
```python
df = pd.read_csv("automobile_price_data3_cleaned.csv")
2. Define Features and Target
The model predicts price based on all other columns.

python
Copy code
X = df.drop(columns=["price"])
y = df["price"]
3. Split Data
Using scikit-learn’s train_test_split:

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
4. Save Output
Two new CSV files are created for easy inspection:

File	Purpose
train_data.csv	Training dataset (80%)
test_data.csv	Test dataset (20%)


# Example 02 — Howell Dataset (Machine Learning Preparation)

This example demonstrates how to **load, inspect, clean, and prepare** the `Howell.csv` dataset for a machine learning project.  
The dataset contains information on individuals’ **height**, **weight**, **age**, and **gender**.

## Step 1: Load and Inspect the Data

The first step loads the dataset safely and performs basic checks for missing values and duplicates.

### Script: `1_data_cleaning.py`

**Purpose:**
- Load `Howell.csv` from the same directory.
- Inspect the data (preview, shape, summary).
- Identify missing values and duplicates.
- Save a cleaned version (`Howell_cleaned.csv`) for later steps.

**Key Code Snippet:**

```python
from pathlib import Path
import pandas as pd
import sys

# Define file path
data_path: Path = Path(__file__).parent / "Howell.csv"

# Check if file exists
if not data_path.exists():
    sys.exit(f"❌ ERROR: Could not find Howell.csv at {data_path}")

# Load CSV with the correct delimiter
howell_df = pd.read_csv(data_path, sep=";")

# Basic checks
print("✅ Howell dataset loaded successfully!")
print(howell_df.head())
print("\nMissing values:\n", howell_df.isnull().sum())
print("\nDuplicate rows:", howell_df.duplicated().sum())
print("\nSummary statistics:\n", howell_df.describe())

# ✅ Optional: Save cleaned version
cleaned_path = Path(__file__).parent / "Howell_cleaned.csv"
howell_df.to_csv(cleaned_path, index=False)
print(f"\n💾 Cleaned file saved to: {cleaned_path}")
💾 Step 1.5: Howell_cleaned.csv
After running the cleaning script, a new file called Howell_cleaned.csv will be created in the same directory:

makefile
Copy code
C:\Repos\applied-ml-foster\notebooks\example02\Howell_cleaned.csv
This cleaned dataset will be used for training and testing in later steps.

# Lab 2 - Exploring and Visualizing the Howell Dataset

This lab uses the cleaned Howell dataset (`Howell_cleaned.csv`) to create visualizations, explore patterns, and add features for analysis.  

Using the cleaned dataset ensures that the visualizations are meaningful and consistent with your workflow.

## Notebook Setup

### Imports
```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable better visuals
sns.set(style="whitegrid")
2️⃣ Load the Cleaned Howell Data
python
Copy code
# Path to your cleaned CSV file
data_path = Path(r"C:\Repos\applied-ml-foster\notebooks\example02\Howell_cleaned.csv")

# Load the cleaned data
howell_df = pd.read_csv(data_path, sep=";")  # change sep="," if needed

# Quick sanity check
howell_df.info()
howell_df.head()
If the file was saved with commas instead of semicolons, change sep=";" to sep=",".

## Quick Visualizations
Height Distribution
Weight Distribution
Height vs Weight by Gender
Create and Visualize BMI
# Calculate BMI
Create BMI Categories
# Check counts
Visualize BMI Category by Gender
Age vs Height (Adults Only)
# Filter for adults (age ≥ 18)
## Splitting the Data by Age and Masking

Sometimes we want to **restrict the data used in plots** without removing any rows. This can be done using **masking**, which tells the plotting function which values to include.  

In this example, we focus on **adult instances only** (`age ≥ 18`) and create masks for **male** and **female** (`male = 1`, `female = 0`).

### Jupyter Notebook Code Example

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Filter for adults (age ≥ 18)
howell_adult = howell_df[howell_df["age"] >= 18]

# Create masks for male and female
male_mask = howell_adult["male"] == 1
female_mask = howell_adult["male"] == 0

# --- Adult Height Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(x=howell_adult["height"][male_mask], kde=True, color="blue", label="Male", alpha=0.6)
sns.histplot(x=howell_adult["height"][female_mask], kde=True, color="pink", label="Female", alpha=0.6)
plt.title("Adult Height Distribution by Gender")
plt.xlabel("Height (cm)")
plt.ylabel("Count")
plt.legend()
plt.show()

# --- Adult Weight Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(x=howell_adult["weight"][male_mask], kde=True, color="blue", label="Male", alpha=0.6)
sns.histplot(x=howell_adult["weight"][female_mask], kde=True, color="pink", label="Female", alpha=0.6)
plt.title("Adult Weight Distribution by Gender")
plt.xlabel("Weight (kg)")
plt.ylabel("Count")
plt.legend()
plt.show()


Lab: Stratified Train/Test Split with Howell Dataset

This lab demonstrates how to prepare the Howell cleaned dataset for machine learning, including:

Calculating BMI and adding categorical features

Filtering adult individuals

Performing a stratified train/test split

Saving the resulting datasets to CSV

Setup
1. Imports
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

2. Load the Cleaned Howell Data
# Path to cleaned CSV
data_path = Path(r"C:\Repos\applied-ml-foster\notebooks\example02\Howell_cleaned.csv")

# Load CSV
howell_df = pd.read_csv(data_path, sep=",")  # CSV uses commas

# Quick sanity check
howell_df.info()
howell_df.head()

3. Add BMI and BMI Categories
# Calculate BMI
howell_df["BMI"] = howell_df["weight"] / (howell_df["height"]/100)**2

# Define BMI categories
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Apply BMI category
howell_df["bmi_category"] = howell_df["BMI"].apply(bmi_category)

# Save updated CSV
howell_df.to_csv(r"C:\Repos\applied-ml-foster\notebooks\example02\Howell_cleaned.csv", index=False)

 Note: In this dataset, only Underweight and Normal categories exist.

4. Filter Adults
# Keep only adults (age ≥ 18)
howell_adult = howell_df[howell_df["age"] >= 18]

5. Prepare Features and Target
X = howell_adult[["height", "weight", "age"]]
y = howell_adult["bmi_category"]

# Remove categories with fewer than 2 samples to avoid stratification errors
valid_categories = y.value_counts()[lambda x: x >= 2].index
mask = y.isin(valid_categories)
X = X[mask]
y = y[mask]

6. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

7. Combine Features and Target & Save CSVs
train_data = X_train.copy()
train_data["bmi_category"] = y_train

test_data = X_test.copy()
test_data["bmi_category"] = y_test

train_data.to_csv(r"C:\Repos\applied-ml-foster\notebooks\example02\train_data.csv", index=False)
test_data.to_csv(r"C:\Repos\applied-ml-foster\notebooks\example02\test_data.csv", index=False)

8. Sanity Check
print("Training set shape:", train_data.shape)
print("Test set shape:", test_data.shape)
print("\nTraining distribution:\n", y_train.value_counts())
print("\nTest distribution:\n", y_test.value_counts())

✅ Notes

Stratified splitting preserves the proportion of BMI categories in both train and test sets.

Only categories with ≥ 2 samples are included to prevent errors.

The final CSV files are ready for downstream modeling tasks.

# Project01 California Housing Price Prediction
## Overview

This project aims to predict the median house values in California's districts using the California Housing dataset. The dataset comprises various features such as median income, average number of rooms, and geographic coordinates. The goal is to build a machine learning model that can accurately estimate house prices based on these features.

Project Workflow
1. Load and Explore the Data

Load the dataset using fetch_california_housing from sklearn.datasets.

Inspect the first few rows to understand the structure of the data.

Check for missing values and handle them appropriately.

Generate summary statistics to grasp the distribution and central tendencies of the features.

2. Visualize Feature Distributions

Histograms: Display the distribution of each numeric feature to understand their spread.

Boxenplots: Identify outliers and visualize the distribution of each feature.

Pairplots: Examine relationships between pairs of features and the target variable.

3. Feature Selection and Target Definition

Select features: Choose relevant features such as 'MedInc' (Median Income) and 'AveRooms' (Average Rooms).

Define the target variable: Set 'MedHouseVal' (Median House Value) as the target.

Prepare the feature matrix (X) and target vector (y) for model training.

4. Train a Linear Regression Model

Split the data: Divide the dataset into training and testing sets (e.g., 80% train, 20% test).

Initialize the model: Create an instance of LinearRegression.

Train the model: Fit the model on the training data.

Make predictions: Use the trained model to predict house prices on the test set.

5. Evaluate the Model

R² (Coefficient of Determination): Measure how well the model explains the variance in the target variable.

MAE (Mean Absolute Error): Calculate the average of the absolute errors between predicted and actual values.

RMSE (Root Mean Squared Error): Compute the square root of the average of squared errors, giving more weight to larger errors.

