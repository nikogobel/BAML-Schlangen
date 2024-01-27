import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from fractions import Fraction
import json
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV

# load the data
file_path_requests = "requests.csv"
file_path_reviews = "reviews.csv"
file_path_recipes = "recipes.csv"
file_path_diet = "diet.csv"

df_requests = pd.read_csv(file_path_requests)
df_reviews = pd.read_csv(file_path_reviews)
df_recipes = pd.read_csv(file_path_recipes)
df_diet = pd.read_csv(file_path_diet)
df_test_reviews = None


def split_reviews():
    global df_test_reviews
    df_test_reviews = df_reviews[df_reviews["Like"].isna()]
    df_test_reviews.dropna(subset=["TestSetId"], inplace=True)
    df_test_reviews.drop("Like", axis=1, inplace=True)
    df_test_reviews.drop("Rating", axis=1, inplace=True)
    df_test_reviews.to_csv("test_reviews.csv")

def clean_reviews():
    global df_reviews
    split_reviews()
    df_reviews.dropna(subset=["Like"], inplace=True)
    df_reviews = df_reviews.drop('Rating', axis=1)
    df_reviews.drop("TestSetId", axis=1, inplace=True)

def clean_diet():
    global df_diet
    df_diet = df_diet.dropna(subset=["Diet"])
    df_diet["Diet"] = df_diet["Diet"].astype("category")
    df_diet["AuthorId"] = df_diet["AuthorId"].astype("string")
    
def translate_recipe_yields_to_categories(s):
    if s == "Undefined":
        return "Undefined"
    result_list = s.split()
    for value in result_list[:1]:
        # print(value)
        if value.find("-") != -1:
            value = s.split("-")[0]
            # print("update")
            # print(value)
        fraction_obj = Fraction(value)
        val_int: float = float(fraction_obj)
        if val_int == 1:
            return "Single Portion"
        elif val_int == 2:
            return "Two Portions"
        elif val_int <= 5:
            return "Medium Portions"
        elif val_int > 5:
            return "Many Portions"
        else:
            return "error"

def clean_recipes():
    global df_recipes
    
    # fill na
    df_recipes["RecipeServings"] = df_recipes["RecipeServings"].fillna(0)

    # set correct type
    df_recipes["Name"] = df_recipes["Name"].astype("string")
    df_recipes["RecipeCategory"] = df_recipes["RecipeCategory"].astype("category")
    df_recipes["RecipeServings"] = df_recipes["RecipeServings"].astype("int64")

    # translate recipe yields
    df_recipes["RecipeYield"] = df_recipes["RecipeYield"].fillna("Undefined")
    df_recipes["RecipeYield"] = df_recipes["RecipeYield"].apply(translate_recipe_yields_to_categories)
    df_recipes["RecipeYield"] = df_recipes["RecipeYield"].astype("category")

    # delete unnecessary columns
    df_recipes = df_recipes.drop('RecipeIngredientParts', axis=1)
    df_recipes = df_recipes.drop('RecipeIngredientQuantities', axis=1)
    df_recipes = df_recipes.drop('Name', axis=1)

    # remove outlier
    max_value_index = df_recipes["PrepTime"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["CookTime"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["Calories"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["FatContent"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["SaturatedFatContent"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["CholesterolContent"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["CholesterolContent"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["CarbohydrateContent"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)
    max_value_index = df_recipes["ProteinContent"].idxmax()
    df_recipes = df_recipes.drop(index=max_value_index).reset_index(drop=True)

def clean_requests():
    global df_requests

    # convert the data types to category
    df_requests["HighCalories"] = df_requests["HighCalories"].astype(int)
    df_requests["HighCalories"] = df_requests["HighCalories"].astype("category")

    df_requests["HighProtein"] = df_requests["HighProtein"].astype("category")

    df_requests["LowFat"] = df_requests["LowFat"].astype("category")

    df_requests["LowSugar"] = df_requests["LowSugar"].astype("category")

    df_requests["HighFiber"] = df_requests["HighFiber"].astype("category")

def rename_columns():
    global df_diet
    global df_recipes
    global df_reviews
    global df_requests

    # rename columns
    df_requests.rename(columns=lambda x: "requests_" + x if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
    df_reviews.rename(columns=lambda x: "reviews_" + x if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
    df_test_reviews.rename(columns=lambda x: "reviews_" + x if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
    df_recipes.rename(columns=lambda x: "recipes_" + x if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
    df_diet.rename(columns=lambda x: "diet_" + x if x not in ["AuthorId", "RecipeId"] else x, inplace=True)

def merge_training_df():
    global df_diet
    global df_recipes
    global df_reviews
    global df_requests

    # join the dataframes
    df_1 = pd.merge(df_diet, df_reviews, on="AuthorId", how="inner")
    df_2 = pd.merge(df_1, df_recipes, on="RecipeId", how="inner")
    df_merged = pd.merge(df_2, df_requests, on=["AuthorId", "RecipeId"], how="inner")

    df_merged = df_merged.drop('AuthorId', axis=1)
    df_merged = df_merged.drop('RecipeId', axis=1)
    df_merged.dropna(inplace=True)

    return df_merged

def merge_test_df():
    global df_diet
    global df_recipes
    global df_requests
    global df_test_reviews

    df_1 = pd.merge(df_diet, df_test_reviews, on="AuthorId", how="inner")
    df_2 = pd.merge(df_1, df_recipes, on="RecipeId", how="inner")
    df_merged = pd.merge(df_2, df_requests, on=["AuthorId", "RecipeId"], how="inner")

    df_merged = df_merged.drop('AuthorId', axis=1)
    df_merged = df_merged.drop('RecipeId', axis=1)
    df_test_set_id = df_merged["reviews_TestSetId"]
    df_test_set_id.to_csv("test_set_id.csv")
    df_merged = df_merged.drop('reviews_TestSetId', axis=1)
    df_merged.dropna(inplace=True)

    return df_merged

def datacleaning():
    seed = 2024
    np.random.seed(seed)

    # clean diet
    print("start cleaning diet")
    clean_diet()
    print("done cleaning diet")

    # clean reviews
    print("start cleaning reviews")
    clean_reviews()
    print("done cleaning reviews")

    # clean recipes
    print("start cleaning recipes")
    clean_recipes()
    print("done cleaning recipes")

    # clean request
    print("start cleaning requests")
    clean_requests()
    print("done cleaning requests")

    # rename columns
    print("start renaming columns")
    rename_columns()
    print("done renaming columns")

    # merge
    print("start merging")
    df_train = merge_training_df()
    df_test = merge_test_df()
    
    non_numeric_cols_train = df_train.select_dtypes(include=['object', 'category']).columns
    non_numeric_cols_test = df_test.select_dtypes(include=['object', 'category']).columns
    
    df_train = pd.get_dummies(df_train, columns=non_numeric_cols_train, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=non_numeric_cols_test, drop_first=True)
    
    print("done merging")

    # save as pickle
    df_train.to_pickle('cleaned_training_dataset.pkl')
    df_train.to_csv('cleaned_training_dataset.csv')
    df_test.to_pickle('cleaned_test_dataset.pkl')
    df_test.to_csv('cleaned_test_dataset.csv')
    print("done saving as pickle")

def load_datasets():
    # Load datasets
    df_train = pd.read_csv('cleaned_training_dataset.csv')
    df_test = pd.read_pickle('cleaned_test_dataset.pkl')
    df_index = pd.read_csv('test_set_id.csv')
    df_submission = pd.read_csv('pub_YwCznU3.csv')
    
    # Drop the 'Unnamed: 0' column
    df_train.drop(columns=['Unnamed: 0'], inplace=True)

    return df_train, df_test, df_index, df_submission

def preprocess_data_NN(df_train, df_test):
    
    df_train = df_train.astype('float32')
    df_test = df_test.astype('float32')
    
    # Splitting the data
    
    X = df_train.drop(columns=['reviews_Like_True'])
    y = df_train['reviews_Like_True'].astype(int)  # Ensure correct encoding
    

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    
 
    # Normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    df_test = df_test.values

    return X_train, y_train, X_val, y_val, df_test

def balanced_accuracy_NN(y_true, y_pred):
    # Cast y_pred to binary (0 or 1)
    y_pred = tf.round(y_pred)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

    # True Positive Rate (TPR) or Sensitivity = TP / (TP + FN)
    tpr = tp / (tp + fn + tf.keras.backend.epsilon())

    # True Negative Rate (TNR) or Specificity = TN / (TN + FP)
    tnr = tn / (tn + fp + tf.keras.backend.epsilon())

    # Balanced Accuracy = (TPR + TNR) / 2
    balanced_acc = (tpr + tnr) / 2

    return balanced_acc

def build_model_NN(input_shape):
    # Neural Network Model
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(64, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(32, kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[balanced_accuracy_NN])

    return model

def train_model_NN(model, X_train, y_train, X_val, y_val):
    # Training the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))

    return history

def evaluate_model_NN(model, X_val, y_val):
    # Evaluating the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    
    
    pred = model.predict(X_val)
    pred = np.round(pred)
    y_val = y_val.to_numpy()
    
    # calculate accuracy on training set
    error_rate = np.mean(y_val != pred)
    print("Error rate:", error_rate)
    print("Balanced Validation Accuracy:", balanced_accuracy_score(y_val, pred))
    
def compute_prediction_NN():
    df_train, df_test, df_index, df_submission= load_datasets()
    X_train, y_train, X_val, y_val, df_test = preprocess_data_NN(df_train, df_test)
    
    model = build_model_NN(input_shape=(X_train.shape[1],))
    
    # train model
    history = train_model_NN(model, X_train, y_train, X_val, y_val)

    evaluate_model_NN(model, X_val, y_val)
    
    # predict on test set
    test_pred = model.predict(df_test)
    
    # match predictions with index
    df_index['test_pred'] = test_pred
    df_submission['prediction_NN'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(
        df_submission['id']).values
    df_submission['prediction_NN'] = df_submission['prediction_NN'].fillna(0.0)
    df_submission['prediction_NN'] = df_submission['prediction_NN'].apply(lambda x: 0 if x < 0.5 else 1).astype(int)
    
    return df_submission

def preprocess_data_RF(df_train, df_test):
    df_train = df_train.astype('float32')
    df_test = df_test.astype('float32')
    
    # Splitting the data
    
    X = df_train.drop(columns=['reviews_Like_True'])
    y = df_train['reviews_Like_True'].astype(int)  # Ensure correct encoding

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    # Normalizing the data
    
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_val, y_val, df_test

def build_model_RF():
    # Random Forest model with default parameters
    model = RandomForestClassifier()

    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [10, 20, 30],
        'max_depth': [2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search to find the best parameters
    grid_search = GridSearchCV(estimator=model, scoring="balanced_accuracy", param_grid=param_grid, cv=5)

    return grid_search

def train_model_RF(grid_search, X_train, y_train):
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model

def evaluate_model_NN(best_model, grid_search, X_train, y_train, X_val, y_val):
    
    pred = best_model.predict(X_train)
    error_rate = np.mean(y_train != pred)
    print("Train Error rate RF:", error_rate)
    print("Train Accuracy RF:", accuracy_score(y_train, pred)) 

    pred_val = best_model.predict(X_val)
    error_rate = np.mean(y_val != pred_val)
    print("Validation Error rate RF:", error_rate)
    print("Validation Accuracy RF:", accuracy_score(y_val, pred_val)) 
    print("Balanced Validation Accuracy RF:", balanced_accuracy_score(y_val, pred_val))
    print("Score on validation set: RF", grid_search.score(X_val, y_val.values.ravel()))

def compute_prediction_RF():
    df_train, df_test, df_index, df_submission= load_datasets()
    X_train, y_train, X_val, y_val, df_test = preprocess_data_RF(df_train, df_test)
    grid_search = build_model_RF()
    
    # train model
    best_model = train_model_RF(grid_search, X_train, y_train)

    evaluate_model_NN(best_model, grid_search, X_train, y_train, X_val, y_val)
    
    # predict on test set
    test_pred = best_model.predict(df_test)
    
    # match predictions with index
    df_index['test_pred'] = test_pred
    df_submission['prediction_RF'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(
        df_submission['id']).values
    df_submission['prediction_RF'] = df_submission['prediction_RF'].fillna(0.0)
    df_submission['prediction_RF'] = df_submission['prediction_RF'].apply(lambda x: 0 if x < 0.5 else 1).astype(int)
    
    return df_submission

def preprocess_data_GDC(df_train, df_test):
    df_train = df_train.astype('float32')
    df_test = df_test.astype('float32')
    
    # Splitting the data
    
    X = df_train.drop(columns=['reviews_Like_True'])
    y = df_train['reviews_Like_True'].astype(int)  # Ensure correct encoding

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    # Normalizing the data
    
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_val, y_val, df_test

def build_model_GDC():
    # Gradient Boosting model with default parameters
    model = GradientBoostingClassifier()

    # Define the parameter grid for grid search
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': ['auto', 'sqrt', 'log2', 0.3, 0.6, 0.9]
    }

    # Perform grid search to find the best parameters
    grid_search = GridSearchCV(estimator=model, scoring="balanced_accuracy", param_grid=param_grid, cv=5)

    return grid_search

def train_model_GDC(grid_search, X_train, y_train):
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model

def evaluate_model_GDC(best_model, grid_search, X_train, y_train, X_val, y_val):
    
    pred = best_model.predict(X_train)
    error_rate = np.mean(y_train != pred)
    print("Train Error rate RF:", error_rate)
    print("Train Accuracy RF:", accuracy_score(y_train, pred)) 

    pred_val = best_model.predict(X_val)
    error_rate = np.mean(y_val != pred_val)
    print("Validation Error rate RF:", error_rate)
    print("Validation Accuracy RF:", accuracy_score(y_val, pred_val)) 
    print("Balanced Validation Accuracy RF:", balanced_accuracy_score(y_val, pred_val))
    print("Score on validation set: RF", grid_search.score(X_val, y_val.values.ravel()))

def compute_prediction_GDC():
    df_train, df_test, df_index, df_submission= load_datasets()
    X_train, y_train, X_val, y_val, df_test = preprocess_data_GDC(df_train, df_test)
    grid_search = build_model_GDC()
    
    # train model
    best_model = train_model_GDC(grid_search, X_train, y_train)
    
    evaluate_model_GDC(best_model, grid_search, X_train, y_train, X_val, y_val)
    
    # predict on test set
    test_pred = best_model.predict(df_test)

    # match predictions with index
    df_index['test_pred'] = test_pred
    df_submission['prediction_GDC'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(
        df_submission['id']).values
    df_submission['prediction_GDC'] = df_submission['prediction_GDC'].fillna(0.0)
    df_submission['prediction_GDC'] = df_submission['prediction_GDC'].apply(lambda x: 0 if x < 0.5 else 1).astype(int)
    
    return df_submission

def preprocess_data_SC(df_train, df_test):
    df_train = df_train.astype('float32')
    df_test = df_test.astype('float32')
    
    # Splitting the data
    
    X = df_train.drop(columns=['reviews_Like_True'])
    y = df_train['reviews_Like_True'].astype(int)  # Ensure correct encoding

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    # Normalizing the data
    
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_val, y_val, df_test

def build_model_SC():
    
    # Define base estimators
    base_estimators = [
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ]
    
    # Define parameter grid
    param_grid = {
        'rf__n_estimators': [10],
        'gb__n_estimators': [10]
    }

    model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())
    

    # Perform grid search to find the best parameters
    grid_search = GridSearchCV(estimator=model, scoring="balanced_accuracy", param_grid=param_grid, cv=5)

    return grid_search

def train_model_SC(grid_search, X_train, y_train):
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model

def evaluate_model_SC(best_model, grid_search, X_train, y_train, X_val, y_val):
    
    pred = best_model.predict(X_train)
    error_rate = np.mean(y_train != pred)
    print("Train Error rate SC:", error_rate)
    print("Train Accuracy SC:", accuracy_score(y_train, pred)) 

    pred_val = best_model.predict(X_val)
    error_rate = np.mean(y_val != pred_val)
    print("Validation Error rate SC:", error_rate)
    print("Validation Accuracy SC:", accuracy_score(y_val, pred_val)) 
    print("Balanced Validation Accuracy SC:", balanced_accuracy_score(y_val, pred_val))
    print("Score on validation set SC:", grid_search.score(X_val, y_val.values.ravel()))
    
def compute_prediction_SC():
    df_train, df_test, df_index, df_submission= load_datasets()
    X_train, y_train, X_val, y_val, df_test = preprocess_data_SC(df_train, df_test)
    grid_search = build_model_SC()
    
    # train model
    best_model = train_model_SC(grid_search, X_train, y_train)
    
    evaluate_model_SC(best_model, grid_search, X_train, y_train, X_val, y_val)
    
    # predict on test set
    test_pred = best_model.predict(df_test)

    # match predictions with index
    df_index['test_pred'] = test_pred
    df_submission['prediction_SC'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(
        df_submission['id']).values
    df_submission['prediction_SC'] = df_submission['prediction_SC'].fillna(0.0)
    df_submission['prediction_SC'] = df_submission['prediction_SC'].apply(lambda x: 0 if x < 0.5 else 1).astype(int)
    
    return df_submission

def main():
    datacleaning()
    #compute_prediction_NN()
    #compute_prediction_RF()
    #compute_prediction_GDC()
    compute_prediction_SC()

if __name__ == "__main__":
    main()
