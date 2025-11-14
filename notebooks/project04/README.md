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
- Use the Titanic dataset from Seaborn and verify its structure.

## Section 2: Data Exploration and Preparation
- Impute missing values for age using the median.
- Drop rows with missing fare.
- Create numeric features such as family_size (sibsp + parch + 1).
- Optionally convert categorical features (e.g., sex, embarked) to numeric.

## Section 3: Feature Selection and Justification
- Create multiple cases for input features:
    - Case 1: age only
    - Case 2: family_size only
    - Case 3: age + family_size
    - Case 4: pclass
- We split each case into training and testing sets for evaluation.
- Age, family size, and passenger class may affect fare because discounts, group size, and class-based pricing influence ticket cost. The dataset includes features such as survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, and alone. Additional features like deck or embark town might improve predictions because they relate to accommodation level or departure location. My Case 4 used one variable, pclass, because passenger class strongly reflects how much someone paid.

## Section 4: Train a Regression Model (Linear Regression)
- Linear Regression
- Ridge Regression
- Elastic Net Regression
- Polynomial Regression (cubic, optionally higher-degree)
- Models are trained using train_test_split and evaluated using R², RMSE, and MAE.
- Overall, Cases 1–3 all underfit because their train and test R² values were very close to zero, showing that age and family size alone did not explain fare well. Case 4 performed best, with train and test scores around 0.30, meaning passenger class explained about 30% of the variation in fare and generalized well without overfitting. Adding age improved the model only slightly, since age is not strongly related to ticket cost—discounts for children or seniors may exist, but they explain only a small amount of variation. Case 1 was the worst model, explaining almost none of the fare variation, and adding more data would not help because age simply isn’t predictive. Case 4 was the strongest, though additional data would offer only limited improvement because class is already a powerful predictor.
-  
## Section 5: Compare Alternative Models (Ridge, Elastic Net, Polynomial Regression)
- Visualize predictions versus actual values, especially for cases with a single input feature.
- Polynomial regression is plotted to examine whether cubic or higher-degree polynomials better capture patterns.
- The cubic model captures the general pattern that fares decrease as passenger class goes from 1st to 3rd, but it struggles with the many outliers in the dataset. It fits best in the dense clusters of points—mainly 2nd and 3rd class—where the data is more consistent, but performs poorly on the extreme fare values. The polynomial model slightly outperforms linear regression by allowing a curved fit through the mid-range of the data, though a straight line would still capture the main trend reasonably well.

## Section 6: Final Thoughts & Insights
- Passenger class was the strongest and most useful feature for predicting fare. Among the models, Elastic Net and polynomial regression performed slightly better than simple linear regression. Adding model complexity or regularization helped reduce overfitting and improved generalization, though the biggest gains came from selecting the right feature (pclass) rather than the model itself. Fare was difficult to predict due to high skew and many outliers—especially in 1st class—and features like age and family size contributed little additional predictive power.

## Optional Next Steps:
- Predict age instead of fare.
- Summary of Findings:
    - Passenger class is a weak predictor of age. All models — Linear, Ridge, Elastic Net, and Polynomial — performed similarly, explaining only about 10–12% of age variation. RMSE stayed around 13 years, showing substantial prediction error. Regularization (Ridge, Elastic Net) did not improve performance, and the cubic polynomial model actually performed slightly worse. Overall, age is not strongly related to passenger class, so no regression method produced meaningful predictive accuracy.
