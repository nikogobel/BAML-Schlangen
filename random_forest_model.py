import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#prepare data
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

df_train_x = df_train.drop(columns=['reviews_Like_True'])
df_train_y = df_train['reviews_Like_True']

X_train, X_test, y_train, y_test = \
  train_test_split(df_train_x, df_train_y,
                   test_size=0.3,
                   shuffle=True,
                   random_state=3)

# data scaling
transform_scaler = StandardScaler()

# dimensionality reduction
transform_pca = PCA()



#fit model
#train_model = RandomForestClassifier(n_estimators=5, max_features=3, random_state=2023+2024)
#train_model.fit(X_train,y_train)

model_logistic_regression = LogisticRegression(max_iter=30)
model_random_forest = RandomForestClassifier()
model_gradient_boosting = GradientBoostingClassifier()
model_first_try_random_forest = RandomForestClassifier()

# train the models
pipeline = Pipeline(steps=[("scaler", transform_scaler),
                           ("pca", transform_pca),
                           ("model", None)])

parameter_grid_preprocessing = {
  "pca__n_components" : [15, 20],
}

parameter_grid_logistic_regression = {
  "model" : [model_logistic_regression],
  "model__C" : [0.1, 10, 20],  # inverse regularization strength
}

parameter_grid_gradient_boosting = {
  "model" : [model_gradient_boosting],
  "model__n_estimators" : [50, 60]
}

parameter_grid_random_forest = {
  "model" : [model_random_forest],
  "model__n_estimators" : [10, 20, 50],  # number of max trees in the forest
  "model__max_depth" : [2, 3, 10],
  "model__max_features" : [3, 5, 10],
  "model__random_state" : [2023+2024]
}

parameter_model_first_try_random_forest = {
  "model" : [model_first_try_random_forest],
  "model__n_estimators" : [5],  # number of max trees in the forest
  "model__max_depth" : [3]
}

meta_parameter_grid = [parameter_grid_logistic_regression,
                       parameter_grid_random_forest,
                       parameter_grid_gradient_boosting,
                       parameter_model_first_try_random_forest]

meta_parameter_grid = [{**parameter_grid_preprocessing, **model_grid}
                       for model_grid in meta_parameter_grid]

search = GridSearchCV(pipeline,
                      meta_parameter_grid,
                      scoring="balanced_accuracy",
                      n_jobs=2,
                      cv=5,  # number of folds for cross-validation
                      error_score="raise"
)

# here, the actual training and grid search happens
search.fit(X_train, y_train.values.ravel())

print("best parameter:", search.best_params_ ,"(CV score=%0.3f)" % search.best_score_)

print("\nOverall model outcomes")
results = search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"Accuracy: {mean_score:.4f}, Parameters: {params}")



#predict on training set
pred = search.best_estimator_.predict(X_train)
error_rate = np.mean(y_train != pred)
print("Train Error rate:", error_rate)
print("Train Accuracy:", accuracy_score(y_train, pred))

#validate the model
pred_val = search.best_estimator_.predict(X_test)
error_rate = np.mean(y_test != pred_val)
print("Validation Error rate:", error_rate)
print("Validation Accuracy:", accuracy_score(y_test, pred_val)) 

print("Score on validation set:", search.score(X_test, y_test.values.ravel()))

# contingency table
print("\nValidation set analysis")
ct = pd.crosstab(pred_val, y_test.values.ravel(),
                 rownames=["pred"], colnames=["true"])
print(ct)
print("\n")

#predict on test set
test_pred = search.best_estimator_.predict(df_test)

#match predictions with index
df_index['test_pred'] = test_pred
df_submission['prediction'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(df_submission['id']).values
#print(df_submission.info())
df_submission['prediction'] = df_submission['prediction'].fillna(False)

#save submission
df_submission['prediction'] = df_submission['prediction'].replace({
    True: 1,
    False: 0
})
df_submission.to_csv('predictions_BAML_Schlangen_1.csv', index=False)

"""

#predict on training set
pred = train_model.predict(X_train)
error_rate = np.mean(y_train != pred)
print("Train Error rate:", error_rate)
print("Train Accuracy:", accuracy_score(y_train, pred))

#validate the model
pred_val = train_model.predict(X_test)
error_rate = np.mean(y_test != pred_val)
print("Validation Error rate:", error_rate)
print("Validation Accuracy:", accuracy_score(y_test, pred_val)) 
"""

