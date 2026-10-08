# Week 4: Machine Learning Model Development and Evaluation
# Breast Cancer Wisconsin Diagnostic Dataset
# Logistic Regression Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

# 1. Load data
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target.map({0: "Malignant", 1: "Benign"})

# 2. Check data
print("Shape:", X.shape)
print("Missing values:", X.isnull().sum().sum())
print("Class distribution:")
print(y.value_counts())

# 3. Split data using stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Build preprocessing + model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42))
])

# 5. Train
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# Probability of malignant class
malignant_index = list(model.classes_).index("Malignant")
y_prob = model.predict_proba(X_test)[:, malignant_index]

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Malignant")
recall = recall_score(y_test, y_pred, pos_label="Malignant")
f1 = f1_score(y_test, y_pred, pos_label="Malignant")
auc_score = roc_auc_score(
    (y_test == "Malignant").astype(int), y_prob
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", auc_score)

# 8. Confusion matrix
cm = confusion_matrix(
    y_test, y_pred, labels=["Malignant", "Benign"]
)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.xticks([0, 1], ["Malignant", "Benign"])
plt.yticks([0, 1], ["Malignant", "Benign"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.show()

# 9. ROC curve
y_true_binary = (y_test == "Malignant").astype(int)
fpr, tpr, _ = roc_curve(y_true_binary, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc_score:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
