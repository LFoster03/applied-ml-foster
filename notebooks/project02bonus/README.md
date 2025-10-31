Iris Dataset Classification Project

Author: Lindsay Foster
Date: 10/30/2025

1. Introduction

The Iris dataset is a classic dataset in machine learning, containing 150 samples of iris flowers from three species: Setosa, Versicolor, and Virginica. Each sample has four features: sepal length, sepal width, petal length, and petal width.

The goal of this project is to build a classification model that predicts the species of an iris flower based on these features. This project follows a structured workflow including data exploration, feature engineering, train/test splitting, modeling, and evaluation.

2. Data Exploration

Dataset Overview

The dataset contains 150 rows and 5 columns (4 features + 1 target).

All features are numeric, and the target is categorical.

Check for Missing Values

The dataset does not have missing values, so no imputation is required.

Unique Values and Class Distribution

There are three unique classes in the target: Setosa, Versicolor, and Virginica.

Class distribution is balanced across the dataset.

Visual Exploration

Pair plots and scatter plots were used to visualize feature relationships and class separation.

Petal length and petal width are particularly useful for distinguishing species.

3. Feature Selection

Input Features: Sepal length, sepal width, petal length, petal width

Target Variable: Species (Setosa, Versicolor, Virginica)

Reflection:

All four numeric features were selected because they provide measurable distinctions between species.

Petal length and width are expected to be highly predictive, while sepal dimensions may provide additional separation.

4. Train/Test Split

The dataset was split into a training set (80%) and a test set (20%).

Stratification was applied to maintain class distribution in both sets.

Reflection:

Stratification helps ensure that each class is proportionally represented in both train and test sets.

Training and test distributions closely match the original dataset, which supports unbiased model evaluation.

5. Feature Scaling (Optional)

Features were optionally standardized to improve the performance of certain classifiers (e.g., Logistic Regression, KNN).

Standardization ensures that all features are on a similar scale.

6. Model Training

Logistic Regression was used as the primary classifier.

The model was trained on the training set and evaluated on the test set.

Reflection:

Logistic Regression is appropriate for multiclass classification and provides interpretable results.

The model leverages numeric feature values to predict species.

7. Model Evaluation

Accuracy, confusion matrix, and classification reports were used to assess model performance.

Reflection:

Setosa is easily separable from other species.

Misclassifications mainly occur between Versicolor and Virginica due to overlapping feature ranges.

Overall, the model achieves high accuracy due to the clear separation of classes in feature space.

8. Feature Importance and Visualization

Decision boundaries and feature plots were created using petal length and petal width.

Visualizations show how well features separate species and help interpret classifier decisions.

Reflection:

Petal measurements are the most informative features for classification.

Visualization highlights areas where the classifier may misclassify overlapping species.

Feature importance analysis supports understanding of the model’s decision-making process.

9. Summary

This project demonstrates a complete workflow for supervised classification using the Iris dataset:

Data exploration and visualization

Feature selection and justification

Train/test splitting with stratification

Optional feature scaling

Model training and evaluation

Interpretation of results through decision boundary visualization

Key Insights:

Petal length and width are highly predictive of species.

Stratified splits preserve class distributions, improving model reliability.

Visual analysis complements numerical evaluation and provides interpretable insights.