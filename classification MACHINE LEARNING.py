import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target  # Adding the target variable
# Discretize the target variable into 3 classes (low, medium, high house prices)
binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')  # You can also try 'uniform'
df['target_binned'] = binner.fit_transform(df[['target']])
# Define features and target
X = df[housing.feature_names]
y = df['target_binned']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Build the classification model (using RandomForestClassifier as an example)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')
# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(clf, X_test_scaled, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
