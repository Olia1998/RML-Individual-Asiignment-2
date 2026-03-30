# RML-Individual-Asiignment-2
Assignment 2 – Model Interpretability and Fairness Analysis
1. Purpose of the Analysis

The purpose of this assignment is to analyze and interpret the predictions of a machine learning model used to assess recidivism risk. The analysis focuses on understanding how different features influence the model’s predictions and identifying potential fairness concerns.

To achieve this, several interpretability methods are used:

LIME (Local Interpretable Model-Agnostic Explanations) to explain individual predictions.
SHAP (SHapley Additive exPlanations) to analyze global and local feature importance.
DiCE (Diverse Counterfactual Explanations) to generate counterfactual scenarios showing how predictions could change.

These techniques help explain the reasoning behind model decisions and allow comparison of explanations across demographic groups. The analysis also evaluates the model’s predictive performance using metrics such as accuracy, precision, recall, false positive rate (FPR), and false negative rate (FNR). Fairness metrics are examined to detect potential disparities in prediction outcomes.

2. Python Libraries Used

The following Python libraries were used in this assignment:

pandas – data preprocessing and manipulation
numpy – numerical operations
matplotlib – visualization of results
seaborn – statistical visualization
statsmodels – logistic regression modeling
scikit-learn – confusion matrix and evaluation metrics
shap – feature attribution explanations
lime – local model explanations
dice-ml – counterfactual explanation generation

Example imports:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import dice_ml
from sklearn.metrics import confusion_matrix
3. Instructions for Reproducing the Results

Follow the steps below to reproduce the results of Assignment 2.

Step 1 – Install required libraries
pip install pandas numpy matplotlib seaborn shap lime dice-ml scikit-learn statsmodels
Step 2 – Open the notebook

Open the Jupyter Notebook file:

Lecture_02_interpretability assignment 2.ipynb
Step 3 – Run the notebook

Run all cells in the notebook sequentially. The notebook performs the following tasks:

Data preprocessing and feature engineering
Logistic regression model training
Model evaluation using a confusion matrix
SHAP global and local explanation plots
LIME explanations for selected predictions
DiCE counterfactual generation
Fairness analysis across racial groups

Executing all cells will reproduce the figures, explanations, and metrics used in the assignment.
