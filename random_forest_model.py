import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

df_train = pd.read_pickle('cleaned_training_dataset.pkl')
df_test = pd.read_pickle('cleaned_test_dataset.pkl')
df_index = pd.read_csv('test_set_id.csv')
df_submission = pd.read_csv('pub_YwCznU3.csv')


# Identify non-numeric columns
non_numeric_cols_train = df_train.select_dtypes(include=['object', 'category']).columns
non_numeric_cols_test = df_test.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to non-numeric columns
df_train = pd.get_dummies(df_train, columns=non_numeric_cols_train, drop_first=True)
df_test = pd.get_dummies(df_test, columns=non_numeric_cols_test, drop_first=True)


train_model = RandomForestClassifier(n_estimators=5, max_features=3, random_state=2023+2024)

#create X and y for training
X_train = df_train.drop(columns=['reviews_Like_True'])
y_train = df_train['reviews_Like_True']

train_model.fit(X_train,y_train)
print("done fitting")

#predict on training set
pred = train_model.predict(X_train)

#calculate accuracy on training set
error_rate = np.mean(y_train != pred)
print("Error rate:", error_rate)
print("Accuracy:", accuracy_score(y_train, pred))

#predict on test set
test_pred = train_model.predict(df_test)

#match predictions with index
df_index['test_pred'] = test_pred
df_submission['prediction'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(df_submission['id']).values
df_submission['prediction'] = df_submission['prediction'].fillna(False)

#save submission
print(df_submission.info())
df_submission.to_csv('submission.csv', index=False)




