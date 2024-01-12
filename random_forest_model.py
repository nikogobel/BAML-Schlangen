import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

df_train = pd.read_pickle('cleaned_training_dataset.pkl')
df_test = pd.read_pickle('cleaned_test_dataset.pkl')


# Identify non-numeric columns
non_numeric_cols = df_train.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to non-numeric columns
df_train = pd.get_dummies(df_train, columns=non_numeric_cols, drop_first=True)

print("Label Balancing in Train Set:\n", df_train['reviews_Like_True'].value_counts(normalize=True))

train_model = RandomForestClassifier(n_estimators=5, max_features=3, random_state=2023+2024)


X_train = df_train.drop(columns=['reviews_Like_True'])
y_train = df_train['reviews_Like_True']

train_model.fit(X_train,y_train)
print("done fitting")

pred = train_model.predict(X_train)

error_rate = np.mean(y_train != pred)
print("Error rate:", error_rate)
print("Accuracy:", accuracy_score(y_train, pred))


