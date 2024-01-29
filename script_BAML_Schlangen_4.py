import pandas as pd
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from joblib import parallel_backend

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
from sklearn.ensemble import StackingClassifier
from fractions import Fraction
import json

from sklearn.svm import SVC

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

seed = 2024
np.random.seed(seed)
    
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
    df_reviews['Rating'] = df_reviews['Rating'].fillna(0)
    df_reviews.drop("TestSetId", axis=1, inplace=True)

    #df_reviews = df_reviews.drop('Rating', axis=1)


# clean diet
def clean_diet():
    global df_diet
    df_diet = df_diet.dropna(subset=["Diet"])
    df_diet["Diet"] = df_diet["Diet"].astype("category")
    df_diet["AuthorId"] = df_diet["AuthorId"].astype("string")


# extract list from
def extract_list_elements(s):
    cleaned_string = s.replace('\\"', '')
    cleaned_string = cleaned_string.replace('c(', '[')
    cleaned_string = cleaned_string.replace(')', ']')
    if s.find("c(") != -1:
        return json.loads(cleaned_string)
    else:
        return []


# extract numbers from string text
def translate_to_int(s):
    if s.find("-") != -1:
        return 0
    result_list = s.split()
    total: float = 0
    for value in result_list:
        fraction_obj = Fraction(value)
        val_int: float = float(fraction_obj)
        if val_int > 0:
            total = total + val_int
    return total


def translate_to_int_loop(s):
    return [float(translate_to_int(element)) for element in s]


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

    # extract lists
    df_recipes['RecipeIngredientParts'] = df_recipes['RecipeIngredientParts'].apply(extract_list_elements)
    df_recipes['RecipeIngredientQuantities'] = df_recipes['RecipeIngredientQuantities'].apply(extract_list_elements)
    df_recipes['RecipeIngredientQuantities'] = df_recipes['RecipeIngredientQuantities'].apply(translate_to_int_loop)

    # get ingredients and how often they are used
    df_ingredients = df_recipes['RecipeIngredientParts']
    df_ingredients = df_ingredients.explode()
    df_ingredients = df_ingredients.drop_duplicates()

    df_count_ingredients = pd.DataFrame(columns=["ingredient", "count"])

    print("this step takes time please wait")
    for search_string in df_ingredients:
        # print(search_string)
        xy = df_recipes['RecipeIngredientParts'].apply(lambda x: x.count(search_string))
        new_entry = {"ingredient": search_string, "count": xy.sum()}
        df_count_ingredients.loc[len(df_count_ingredients)] = new_entry

    df_ingredients_sorted = df_count_ingredients.sort_values(by='count', ascending=False)

    print("extracting most important ingredients")
    # safe 10 most important ingedrients in extra columns
    for i, r in df_ingredients_sorted.head(5).iterrows():
        search_string = r['ingredient']
        df_recipes[f'Recipe_IngredientParts_{search_string}'] = 0

        for index, row in df_recipes.iterrows():
            if search_string in row['RecipeIngredientParts']:
                search_index = row['RecipeIngredientParts'].index(search_string)
                try:
                    new_value = row['RecipeIngredientQuantities'][search_index]
                    df_recipes.at[index, f'Recipe_IngredientParts_{search_string}'] = new_value
                except:
                    continue

    print("done with ingredients")
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
    #df_recipes = df_recipes.drop('RecipeYield', axis=1)

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

    #df_requests = df_requests.drop('Time', axis=1)


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
    df_training = merge_training_df()
    df_test = merge_test_df()
    print("done merging")

    # save as pickle
    df_training.to_pickle('cleaned_training_dataset.pkl')
    df_training.to_csv('cleaned_training_dataset.csv')
    df_test.to_pickle('cleaned_test_dataset.pkl')
    df_test.to_csv('cleaned_test_dataset.csv')
    print("done saving as pickle")

    # check results
    df_training = pd.read_pickle('cleaned_training_dataset.pkl')
    print(df_training.head())
    print(df_training.info())


def model():
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

    df_train = df_train.drop('recipes_Recipe_IngredientParts_butter', axis=1)
    df_test = df_test.drop('recipes_Recipe_IngredientParts_butter', axis=1)

    df_train = df_train.drop('recipes_Recipe_IngredientParts_onion', axis=1)
    df_test = df_test.drop('recipes_Recipe_IngredientParts_onion', axis=1)

    #adjust scaling
    columns_to_log_scale = [
        #'reviews_Rating',
        'recipes_SugarContent',
        'recipes_FiberContent',
        'recipes_CarbohydrateContent',
        'recipes_SodiumContent',
        'recipes_CholesterolContent',
        #'recipes_SaturatedFatContent',
        #'recipes_FatContent',
        #'recipes_PrepTime',
        #'recipes_CookTime'
    ]

    #df_train[columns_to_log_scale] = np.where(df_train[columns_to_log_scale] > 0, np.log(df_train[columns_to_log_scale]), df_train[columns_to_log_scale])
    #df_train[columns_to_log_scale] = df_train[columns_to_log_scale].astype(float)
    #df_test[columns_to_log_scale] = np.where(df_test[columns_to_log_scale] > 0, np.log(df_test[columns_to_log_scale]), df_test[columns_to_log_scale])
    #df_test[columns_to_log_scale] = df_test[columns_to_log_scale].astype(float)

    # Identify non-numeric columns
    non_numeric_cols_train = df_train.select_dtypes(include=['object', 'category']).columns
    non_numeric_cols_test = df_test.select_dtypes(include=['object', 'category']).columns

    df_train = pd.get_dummies(df_train, columns=non_numeric_cols_train, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=non_numeric_cols_test, drop_first=True)

    train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['reviews_Like_True'])

    #only inside interval
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

    #for column in columns_quantile_removal:
    #    lower_quantile = train_set[column].quantile(0.01)
    #    upper_quantile = train_set[column].quantile(0.99)
    #    train_set = train_set[(train_set[column] >= lower_quantile) & (train_set[column] <= upper_quantile)]


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

    # data scaling
    transform_scaler = StandardScaler()

    # dimensionality reduction
    transform_pca = PCA()



    #fit model
    model_logistic_regression = LogisticRegression(max_iter=30)
    model_random_forest = RandomForestClassifier()
    model_gradient_boosting = GradientBoostingClassifier()
    model_first_try_random_forest = RandomForestClassifier()
    model_balanced_random_forest = BalancedRandomForestClassifier()
    svm_model = SVC(random_state=42)
    rf_stacking = RandomForestClassifier(max_depth=10, n_estimators=30, max_features=10, random_state=2023+2024, min_samples_leaf=1)
    gb_stacking = GradientBoostingClassifier(learning_rate=0.3, min_samples_leaf=2, min_samples_split=2, n_estimators=200)
    model_stacking = StackingClassifier(estimators=[('rf', rf_stacking), ('gb', gb_stacking)], final_estimator=model_logistic_regression())

    # train the models
    pipeline = Pipeline(steps=[#("scaler", transform_scaler),
                            #("pca", transform_pca),
                            ("model", None)])

    parameter_grid_preprocessing = {
    "pca__n_components" : [15, 20],
    }

    parameter_grid_logistic_regression = {
    "model" : [model_logistic_regression],
    "model__C" : [0.1, 10],  # inverse regularization strength
    }

    parameter_grid_gradient_boosting = {
    "model" : [model_gradient_boosting],
    "model__n_estimators" : [200],
    "model__learning_rate" : [0.3],
    "model__min_samples_leaf" : [2],
    "model__min_samples_split" : [2]
    }

    parameter_grid_svc = {
    "model" : [svm_model],
    }

    parameter_grid_random_forest = {
    "model" : [model_random_forest],
    "model__n_estimators" : [30],  # number of max trees in the forest
    "model__max_depth" : [10],
    "model__max_features" : [10],
    "model__min_samples_leaf" : [1],#3
    #"model__min_samples_split" : [2],
    "model__random_state" : [2023+2024]
    }

    parameter_model_first_try_random_forest = {
    "model" : [model_first_try_random_forest],
    "model__n_estimators" : [5],  # number of max trees in the forest
    "model__max_depth" : [3]
    }

    paramezer_model_balanced_random_forest = {
    "model" : [model_balanced_random_forest],
    "model__n_estimators" : [5],  # number of max trees in the forest
    "model__max_depth" : [3]
    }
    
    paramter_model_stacking

    meta_parameter_grid = [#parameter_grid_logistic_regression,
                        #parameter_grid_random_forest,
                        parameter_grid_gradient_boosting
                            #parameter_grid_svc
                        ]
                        #parameter_model_first_try_random_forest,
                        #paramezer_model_balanced_random_forest]

    meta_parameter_grid = [{**model_grid} #{**parameter_grid_preprocessing, **model_grid}
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
    print(df_submission['prediction'].value_counts())
    df_submission.to_csv('predictions_BAML_Schlangen_6.csv', index=False)

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

def main():
    datacleaning()
    model()

if __name__ == "__main__":
    main()
