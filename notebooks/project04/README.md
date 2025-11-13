# Project 4 – Predicting a Continuous Target with Regression: Titanic Dataset
**Name:** Lindsay Foster 
**Date:** 11/11/2025

## Introduction
This project focuses on regression modeling using the Titanic dataset. This project will predict fare and the amount of money paid for the journey. This will also predict a continuous numeric target.

# Steps

- Create the virtual environment

- Use Python’s built-in venv module: python -m venv .venv. .venv is the folder where your virtual environment will be stored.After this, your project folder will have a .venv directory.Tip: Use lowercase .venv (or venv) for convention. VS Code automatically recognizes .venv.

- Activate the virtual environment: .venv\Scripts\Activate. A requirements.txt lists all the Python packages your project needs so others (or you in a new environment) can install them easily with pip install -r requirements.txt.

- How to use it - Save this file as requirements.txt in your project folder. Make sure your virtual environment is activated (.venv). Install all packages with: pip install -r requirements.txt. Select .venv as the kernel. Tell VS Code to use this .venv interpreter. Press Ctrl+Shift+P → type Python: Select Interpreter. Look for the interpreter in your .venv, e.g.:C:\Repos\project02\.venv\Scripts\python.exe. Select it. Reload VS Code to make sure it takes effect. This makes your editor aware of all packages installed in .venv.

## Project 4 Steps
- Opening with title, name or alias, date, and short intro describing dataset and objectives. 
- Section 1: Import and Inspect the Data
- Section 2: Data Exploration and Preparation
- Section 3: Feature Selection and Justification
- Section 4: Train a Regression Model (Linear Regression)
- Section 5: Compare Alternative Models (Ridge, Elastic Net, Polynomial Regression)
- Section 6: Final Thoughts & Insights


## Section 1: Import and Inspect the Data
We use the Titanic dataset from Seaborn and verify its structure.

## Section 2: Data Exploration and Preparation
Impute missing values for age using the median.

Drop rows with missing fare.

Create numeric features such as family_size (sibsp + parch + 1).

Optionally convert categorical features (e.g., sex, embarked) to numeric.

## Section 3: Feature Selection and Justification
We create multiple cases for input features:

Case 1: age only

Case 2: family_size only

Case 3: age + family_size

Case 4: pclass (or another chosen variable)

We split each case into training and testing sets for evaluation.

## Section 4: Train a Regression Model (Linear Regression)
Linear Regression

Ridge Regression

Elastic Net Regression

Polynomial Regression (cubic, optionally higher-degree)

Models are trained using train_test_split and evaluated using R², RMSE, and MAE.

## Section 5: Compare Alternative Models (Ridge, Elastic Net, Polynomial Regression)
We visualize predictions versus actual values, especially for cases with a single input feature.
Polynomial regression is plotted to examine whether cubic or higher-degree polynomials better capture patterns.

## Section 6: Final Thoughts & Insights
Most useful features: pclass often provides the strongest predictive power.

Best-performing model: Depends on the input features; Elastic Net or Polynomial regression sometimes outperform simple linear regression.

Effect of complexity/regularization: Regularization reduces overfitting and improves generalization.

Challenges:

Fare is highly skewed with outliers, making it hard to predict.

Some features have weak correlation with fare.

Optional Next Steps:

Predict age instead of fare.

Explore log transformation of fare to reduce skew.

Include additional features such as sex or embarked.
