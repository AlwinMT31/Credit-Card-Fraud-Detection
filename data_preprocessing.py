import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("cleaned_creditcard_data.csv")

# Explore dataset
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print(df['IsFraud'].value_counts())

# Visualize class distribution
sns.countplot(x='IsFraud', data=df)
plt.title("Legit vs Fraud")
plt.show()

# Feature scaling
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Train-test split
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
