# fraud_detection.py

# ========== Imports ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# ========== Load & Explore Data ==========
df = pd.read_csv("cleaned_creditcard_data.csv")

print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Class distribution:\n", df['IsFraud'].value_counts())

# Visualize class imbalance
sns.countplot(x='IsFraud', data=df)
plt.title("Legitimate vs Fraudulent Transactions")
plt.xticks([0, 1], ['Legit (0)', 'Fraud (1)'])
plt.show()

# ========== Preprocessing ==========
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== Model Training ==========
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# ========== Evaluation ==========
y_pred = xgb_model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred))

RocCurveDisplay.from_estimator(xgb_model, X_test, y_test)
plt.title("ROC Curve - XGBoost")
plt.show()

# ========== Save Model ==========
joblib.dump(xgb_model, 'random_forest_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved successfully.")
