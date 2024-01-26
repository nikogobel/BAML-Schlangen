import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC

np.random.seed(42)
#prepare data
df_train = pd.read_pickle('cleaned_training_dataset.pkl')
df_test = pd.read_pickle('cleaned_test_dataset.pkl')
df_index = pd.read_csv('test_set_id.csv')
df_submission = pd.read_csv('pub_YwCznU3.csv')

#remove col
df_train = df_train.drop('recipes_RecipeYield', axis=1)
df_test = df_test.drop('recipes_RecipeYield', axis=1)

df_train = df_train.drop('requests_Time', axis=1)
df_test = df_test.drop('requests_Time', axis=1)

df_train = df_train.drop('reviews_Rating', axis=1)
df_test = df_test.drop('reviews_Rating', axis=1)

#only required if ingredients were extracted
df_train = df_train.drop('recipes_Recipe_IngredientParts_butter', axis=1)
df_test = df_test.drop('recipes_Recipe_IngredientParts_butter', axis=1)

df_train = df_train.drop('recipes_Recipe_IngredientParts_onion', axis=1)
df_test = df_test.drop('recipes_Recipe_IngredientParts_onion', axis=1)

#adjust scaling to log for the following variables
columns_to_log_scale = [
    'recipes_SugarContent',
    'recipes_FiberContent',
    'recipes_CarbohydrateContent',
    'recipes_SodiumContent',
    'recipes_CholesterolContent',
    'recipes_SaturatedFatContent',
    'recipes_FatContent',
    'recipes_PrepTime',
    'recipes_CookTime'
]

df_train[columns_to_log_scale] = np.where(df_train[columns_to_log_scale] > 0, np.log(df_train[columns_to_log_scale]), df_train[columns_to_log_scale])
df_train[columns_to_log_scale] = df_train[columns_to_log_scale].astype(float)
df_test[columns_to_log_scale] = np.where(df_test[columns_to_log_scale] > 0, np.log(df_test[columns_to_log_scale]), df_test[columns_to_log_scale])
df_test[columns_to_log_scale] = df_test[columns_to_log_scale].astype(float)

# Identify non-numeric columns
non_numeric_cols_train = df_train.select_dtypes(include=['object', 'category']).columns
non_numeric_cols_test = df_test.select_dtypes(include=['object', 'category']).columns

df_train = pd.get_dummies(df_train, columns=non_numeric_cols_train, drop_first=True)
df_test = pd.get_dummies(df_test, columns=non_numeric_cols_test, drop_first=True)

train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['reviews_Like_True'])

#take only samples where all values of the following columns are inside interval
#currently very large, because otherwise too many samples are removed
columns_quantile_removal = [
    'recipes_CookTime',
    'recipes_PrepTime',
    'recipes_Calories',
    'recipes_FatContent',
    'recipes_SaturatedFatContent',
    'recipes_CholesterolContent',
    'recipes_SodiumContent',
    'recipes_CarbohydrateContent',
    'recipes_FiberContent',
    'recipes_SugarContent',
    'recipes_ProteinContent',
    'recipes_RecipeServings',
    'recipes_Recipe_IngredientParts_salt',
    'recipes_Recipe_IngredientParts_sugar',
    'recipes_Recipe_IngredientParts_water'
]

for column in columns_quantile_removal:
    lower_quantile = train_set[column].quantile(0.01)
    upper_quantile = train_set[column].quantile(0.99)
    train_set = train_set[(train_set[column] >= lower_quantile) & (train_set[column] <= upper_quantile)]


#4-fache Anzahl an samples mit like = true
df_duplicated_y1 = pd.concat([train_set[train_set['reviews_Like_True'] == True]] * 4, ignore_index=True)
train_set = pd.concat([train_set, df_duplicated_y1], ignore_index=True)

#train predictions and explanatory variabales
X_train = train_set.drop('reviews_Like_True', axis=1)
y_train = train_set['reviews_Like_True']

X_test = test_set.drop('reviews_Like_True', axis=1)
y_test = test_set['reviews_Like_True']

print("Number of samples with y=0 in y_train:", y_train.value_counts()[0])
print("Number of samples with y=1 in y_train:", y_train.value_counts()[1])

#fit model
model_random_forest = RandomForestClassifier()
model_gradient_boosting = GradientBoostingClassifier()

# train the models
pipeline = Pipeline(steps=[("model", None)])

parameter_grid_gradient_boosting = {
  "model" : [model_gradient_boosting],
  "model__n_estimators" : [200],
  "model__learning_rate" : [0.3],
  "model__min_samples_leaf" : [2],
  "model__min_samples_split" : [2]
}

parameter_grid_random_forest = {
  "model" : [model_random_forest],
  "model__n_estimators" : [30],  # number of max trees in the forest
  "model__max_depth" : [9],
  "model__max_features" : [10],
  "model__min_samples_leaf" : [3],
  "model__min_samples_split" : [2],
  "model__random_state" : [2023+2024]
}

meta_parameter_grid = [parameter_grid_random_forest,
                       parameter_grid_gradient_boosting
                       ]

meta_parameter_grid = [{**model_grid}
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
print("Balanced Validation Accuracy:", balanced_accuracy_score(y_test, pred_val))
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

