# Week-4-Machine-Learning-Model-Development-and-Evaluation
# Week 4: Machine Learning Model Development and Evaluation

## Project Overview

This project demonstrates a complete basic machine learning workflow using Python and Scikit-learn. The objective is to develop and evaluate a classification model for predicting whether a breast tumor is **malignant or benign**.

The **Breast Cancer Wisconsin (Diagnostic) dataset** is used for this project. The dataset contains 569 observations and 30 numerical features related to breast tissue measurements.

## Objectives

* Prepare and inspect the dataset
* Check for missing values
* Split the data into training and testing sets
* Scale numerical features
* Develop a Logistic Regression model
* Evaluate model performance
* Visualize classification results
* Analyze errors and model limitations
* Suggest possible improvements

## Methodology

The dataset was divided into **80% training data and 20% testing data** using stratified sampling. `StandardScaler` was used for feature scaling, followed by a **Logistic Regression** classifier.

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* ROC Curve

## Results

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 96.5% |
| Precision | 97.5% |
| Recall    | 92.9% |
| F1-score  | 95.1% |
| ROC-AUC   | 99.6% |

## Visualizations

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### ROC Curve

![ROC Curve](roc_curve.png)

### Performance Metrics

![Performance Metrics](model_metrics.png)

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Google Colab

## Repository Contents

* `week4_machine_learning.py` — Python implementation
* `Week4_Machine_Learning.ipynb` — Jupyter Notebook
* `Week4_Machine_Learning_Report.docx` — Detailed project report
* `confusion_matrix.png` — Confusion matrix
* `roc_curve.png` — ROC curve
* `model_metrics.png` — Performance comparison
* `test_predictions.csv` — Model predictions
* `requirements.txt` — Required Python libraries

## Limitations and Future Improvements

The project uses a single train-test split and only one machine learning algorithm. Future work can include k-fold cross-validation, hyperparameter tuning, and comparison with Random Forest, Support Vector Machine, and Decision Tree models. Testing on an independent external dataset would also help evaluate generalization.

> **Note:** This is an educational machine learning project and is not intended for clinical diagnosis.

## Author

**Tirtha Das**
Biotechnology Student
