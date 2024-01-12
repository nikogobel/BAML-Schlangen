import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_pickle('cleaned_dataset.pkl')

#print(df.info())
#print(df.head())

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)
df = df.dropna()

df.to_csv('cleaned_dataset_noNA.csv')

print("Label Balancing in Train Set:\n", df['reviews_Like_True'].value_counts(normalize=True))

train_model = RandomForestClassifier(n_estimators=5, max_features=3, random_state=2023+2024)


X = df.drop(columns=['reviews_Like_True'])
y = df['reviews_Like_True']



train_model.fit(X,y)
print("done fitting")

pred = train_model.predict(X)

error_rate = np.mean(y != pred)
print("Error rate:", error_rate)
print("Accuracy:", accuracy_score(y, pred))
