# Project 5 – Ensemble Machine Learning on the Wine Quality Dataset
**Name:** Lindsay Foster 
**Date:** 11/19/2025

## Introduction
In this project, we explore ensemble machine learning techniques to improve predictive performance on the Wine Quality Dataset from the UCI Machine Learning Repository. Traditional single models, such as decision trees or logistic regression, may struggle with complex, nonlinear relationships in data. Ensemble models solve this problem by combining multiple learners to reduce overfitting, increase stability, and improve accuracy.

We focus on three major ensemble approaches:

Random Forests – many decision trees trained in parallel using bootstrap sampling.

Gradient Boosting – sequential trees that correct the mistakes of earlier models.

Voting Classifier – a heterogeneous ensemble that combines different model types using majority vote or probability averaging.

To evaluate our models, we use comprehensive performance metrics including accuracy, precision, recall, and F1 score, which are especially important for multi-class classification like wine quality (low, medium, high). We also evaluate how well our models generalize by examining the gap between training and testing metrics.

By the end of this project, we will determine which ensemble model performs best, which generalizes most reliably, and how ensemble strategies compare to simpler machine learning models.

# Steps

- Create the virtual environment

- Use Python’s built-in venv module: python -m venv .venv. .venv is the folder where your virtual environment will be stored.After this, your project folder will have a .venv directory.Tip: Use lowercase .venv (or venv) for convention. VS Code automatically recognizes .venv.

- Activate the virtual environment: .venv\Scripts\Activate. A requirements.txt lists all the Python packages your project needs so others (or you in a new environment) can install them easily with pip install -r requirements.txt.

- How to use it - Save this file as requirements.txt in your project folder. Make sure your virtual environment is activated (.venv). Install all packages with: pip install -r requirements.txt. Select .venv as the kernel. Tell VS Code to use this .venv interpreter. Press Ctrl+Shift+P → type Python: Select Interpreter. Look for the interpreter in your .venv, e.g.:C:\Repos\project02\.venv\Scripts\python.exe. Select it. Reload VS Code to make sure it takes effect. This makes your editor aware of all packages installed in .venv.

## Project 5 Steps
Save csv into data folder: 
```
import pandas as pd

# Download the CSV from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Load into a DataFrame (semicolon-separated!)
df = pd.read_csv(url, sep=";")

# Save to your data folder
save_path = r"C:\Repos\applied-ml-foster\notebooks\project05\data\winequality-red.csv"
df.to_csv(save_path, index=False)

df.head()
```
## Step 1: Load and Inspect the Data

Dataset contains 1599 samples, 11 features (e.g., acidity, sulphates, alcohol) and 1 target (quality rated 0–10).

Simplified quality into three categories: low (3–4), medium (5–6), high (7–8).

Checked column types, missing values, and initial distribution.

## Step 2: Prepare the Data

Created quality_label (string) and quality_numeric (0=low, 1=medium, 2=high) columns for modeling.

Applied helper functions to map numeric ratings to categories.

## Step 3: Feature Selection

Features (X): all columns except quality, quality_label, quality_numeric.

Target (y): quality_numeric.

## Step 4: Split the Data

Train/test split 80/20 with stratification to preserve class balance.

## Step 5: Train and Evaluate Models

Models tested:

    - Random Forest (100 trees)

    - Gradient Boosting (100 trees)

    - Voting Classifier (DT + SVM + NN)

Evaluated using Accuracy, Precision, Recall, F1 Score on both train and test sets.

Calculated train-test gaps to assess overfitting.

## Step 6: Compare Results

**Model Performance Comparison**

| Model                           | Train Accuracy | Test Accuracy | Train F1 | Test F1 | Accuracy Gap | F1 Gap |
|---------------------------------|----------------|---------------|----------|---------|--------------|--------|
| Random Forest (100 trees)       | 0.92           | 0.89          | 0.90     | 0.87    | 0.03         | 0.03   |
| Voting Classifier (DT+SVM+NN)   | 0.91           | 0.87          | 0.88     | 0.84    | 0.04         | 0.04   |
| Gradient Boosting (100 trees)   | 0.89           | 0.86          | 0.88     | 0.84    | 0.03         | 0.04   |


Random Forest had highest test accuracy but slightly higher overfitting.

Voting Classifier had smaller train/test gap → better generalization.

Gradient Boosting performed slightly lower but still competitive.

## Step 7: Conclusions and Insights

Random Forest is recommended for accuracy, while Voting Classifier balances generalization and stability.

Feature importance highlights alcohol, acidity, and sulphates as key predictors.

Ensembles clearly outperform single models and provide robust predictions for red wine quality.