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
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2, l1_l2

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


# clean reviews
def split_reviews():
    global df_test_reviews
    df_test_reviews = df_reviews[df_reviews["Like"].isna()]
    df_test_reviews.dropna(subset=["TestSetId"], inplace=True)
    df_test_reviews.drop("Like", axis=1, inplace=True)
    df_test_reviews['Rating'] = df_test_reviews['Rating'].fillna(0)
    df_test_reviews.to_csv("test_reviews.csv")


def clean_reviews():
    global df_reviews
    split_reviews()
    df_reviews.dropna(subset=["Like"], inplace=True)
    df_reviews = df_reviews.drop('Rating', axis=1)
    df_reviews.drop("TestSetId", axis=1, inplace=True)


# clean diet
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

    return df_train, df_test, df_index, df_submission

def preprocess_data(df_train, df_test):
    
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

    return X_train, y_train, X_val, y_val, df_test

def build_model(input_shape):
    # Neural Network Model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_val, y_val):
    # Training the model
    history = model.fit(X_train, y_train, epochs=60, batch_size=64, validation_data=(X_val, y_val))

    return history

def evaluate_model(model, X_val, y_val):
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
    
def compute_prediction():
    df_train, df_test, df_index, df_submission= load_datasets()
    X_train, y_train, X_val, y_val, df_test = preprocess_data(df_train, df_test)
    model = build_model(input_shape=(X_train.shape[1],))
    
    # train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    evaluate_model(model, X_val, y_val)
    
    # predict on test set
    test_pred = model.predict(df_test)
    
    # match predictions with index
    df_index['test_pred'] = test_pred
    df_submission['prediction'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(
        df_submission['id']).values
    df_submission['prediction'] = df_submission['prediction'].fillna(0.0)
    df_submission['prediction'] = df_submission['prediction'].apply(lambda x: 0 if x < 0.5 else 1).astype(int)
    
    # save submission
    df_submission.to_csv('predictions_BAML_Schlangen_2.csv', index=False)

def main():
    datacleaning()
    compute_prediction()

if __name__ == "__main__":
    main()
