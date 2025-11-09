# Project 3 – Titanic Dataset Analysis
**Name:** Lindsay Foster 
**Date:** 11/06/2025

## Introduction
In this project, we use the Titanic dataset to build and evaluate three classifiers: Decision Tree, Support Vector Machine, and Neural Network. We compare model performance across three different feature sets and reflect on their effectiveness for predicting passenger survival.

# Steps

- Create the virtual environment

- Use Python’s built-in venv module: python -m venv .venv. .venv is the folder where your virtual environment will be stored.After this, your project folder will have a .venv directory.Tip: Use lowercase .venv (or venv) for convention. VS Code automatically recognizes .venv.

- Activate the virtual environment: .venv\Scripts\Activate.ps1. A requirements.txt lists all the Python packages your project needs so others (or you in a new environment) can install them easily with pip install -r requirements.txt.

### For Project 3 – Titanic Dataset Analysis, here’s a typical requirements.txt you can use:

pandas==2.2.0
numpy==1.26.0
matplotlib==3.8.1
seaborn==0.12.2
scikit-learn==1.3.2
jupyter==1.0.0
notebook==7.8.0

- How to use it - Save this file as requirements.txt in your project folder. Make sure your virtual environment is activated (.venv). Install all packages with: pip install -r requirements.txt. Select .venv as the kernel. Tell VS Code to use this .venv interpreter. Press Ctrl+Shift+P → type Python: Select Interpreter. Look for the interpreter in your .venv, e.g.:C:\Repos\project02\.venv\Scripts\python.exe. Select it. Reload VS Code to make sure it takes effect. This makes your editor aware of all packages installed in .venv.

## Project 3 – Titanic Dataset Analysis

### Steps:
1. Import and Inspect Data
2. Handle Missing Values and Clean Data
3. Create New Features
4. Choose Feature and Target
5. Define X (features) and y (target)
6. Train the Classification Model - Split the Data
7. Create and Train Model (Decision Tree)
8. Predict and Evaluate Model Performance
9. Report Confusion Matrix (as a heatmap)
10. Report Decision Tree Plot
11. Compare Alternative Models
12. Train and Evaluate Model (SVC)
13. Visualize Support Vectors
14. Train and Evaluate Model (Neural Network)
15. Visualize
16. Markdown Summary Table

| Model Type | Case | Features Used | Accuracy | Precision | Recall | F1-Score | Notes |
|-------------|------|---------------|----------|------------|--------|-----------|-------|
| Decision Tree | Case 1 | alone | 63.00% | 64.00% | 63.00% | 63.00% | Moderate accuracy — being alone influenced survival but not strongly enough for high predictive power. |
|               | Case 2 | age | 61.00% | 58.00% | 61.00% | 55.00% | Age shows weak predictive capability. Older passengers had lower survival, but the model couldn’t generalize well. |
|               | Case 3 | age + family_size | 59.00% | 57.00% | 59.00% | 57.00% | Adding family size didn’t improve accuracy; it might have introduced noise or redundancy. |
| **SVM (RBF Kernel)** | Case 1 | alone | 63.00% | 64.00% | 63.00% | 63.00% | default kernel |
|                     | Case 2 | age | 63.00% | 66.00% | 63.00% | 52.00% | - |
|                     | Case 3 | age + family_size | 63.00% | 66.00% | 63.00% | 62.00% | - |
| Neural Network (MLP) | Case 1 | alone | 63.00% | 64.00% | 63.00% | 63.00% | - |
|                     | Case 2 | age | 63.00% | 69.00% | 63.00% | 51.00% | - |
|                     | Case 3 | age + family_size | 61.00% | 38.00% | 61.00% | 47.00% | - |
