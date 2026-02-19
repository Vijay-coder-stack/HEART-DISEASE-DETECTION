# =====================================================
# HEART DISEASE DETECTION STREAMLIT APP
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

st.set_page_config(page_title="Heart Disease Detection", layout="wide")

st.title("❤️ Heart Disease Detection using Classification Algorithms")
st.write("Dataset: heart_disease_dataset.csv")

# =====================================================
# LOAD DATASET
# =====================================================

try:
    data = pd.read_csv("heart_disease_dataset")
except FileNotFoundError:
    st.error("❌ Dataset file not found.")
    st.stop()

# =====================================================
# SPLIT FEATURES & TARGET
# =====================================================

X = data.drop("heart_disease", axis=1)
y = data["heart_disease"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# TRAIN MODELS
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Model": model,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

# =====================================================
# DISPLAY MODEL COMPARISON
# =====================================================

st.subheader("📊 Model Comparison")

results_df = pd.DataFrame(results).T
st.dataframe(results_df)

# Select best model (Highest ROC-AUC)
best_model_name = results_df["ROC-AUC"].idxmax()
best_model = results[best_model_name]["Model"]

st.success(f"Best Model: {best_model_name}")

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("🔍 Confusion Matrix (Best Model)")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# =====================================================
# PREDICTION SECTION
# =====================================================

st.subheader("🩺 Predict Heart Disease")

input_data = []

for col in X.columns:
    value = st.number_input(
        f"{col}",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    input_data.append(value)

if st.button("Predict"):

    input_array = np.array([input_data])
    prediction = best_model.predict(input_array)[0]
    probability = best_model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {probability:.2f})")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("Developed using Streamlit | Logistic Regression | Decision Tree | Random Forest | SVM")
