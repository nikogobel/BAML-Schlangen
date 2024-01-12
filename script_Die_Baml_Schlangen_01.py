import pandas as pd
import numpy as np
from fractions import Fraction
import json


# load the data
file_path_requests = "requests.csv"
file_path_reviews = "reviews.csv"
file_path_recipes = "recipes.csv"
file_path_diet = "diet.csv"

df_requests = pd.read_csv(file_path_requests)
df_reviews = pd.read_csv(file_path_reviews)
df_recipes = pd.read_csv(file_path_recipes)
df_diet = pd.read_csv(file_path_diet)


#clean diet
def clean_diet():
    global df_diet
    df_diet = df_diet.dropna(subset=["Diet"])
    df_diet["Diet"] = df_diet["Diet"].astype("category")
    df_diet["AuthorId"] = df_diet["AuthorId"].astype("string")
    """ df_diet.rename(columns={
        'Diet': 'Author_Diet',
        'Age': 'Author_Age',
    }, inplace=True)
    return df_diet """


#extract list from
def extract_list_elements(s):
    cleaned_string = s.replace('\\"', '')
    cleaned_string = cleaned_string.replace('c(', '[')
    cleaned_string = cleaned_string.replace(')', ']')
    if s.find("c(") != -1:
        return json.loads(cleaned_string)
    else:
        return []


#extract numbers from string text
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
        #print(value)
        if value.find("-") != -1:
            value = s.split("-")[0]
            #print("update")
            #print(value)
        fraction_obj = Fraction(value)
        val_int: float = float(fraction_obj)
        if val_int == 1:
            return "Single Portion"
        elif val_int == 2:
            return "Two Portions"
        elif val_int <=5:
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
        #print(search_string)
        xy = df_recipes['RecipeIngredientParts'].apply(lambda x: x.count(search_string))
        new_entry = {"ingredient": search_string, "count": xy.sum()}
        df_count_ingredients.loc[len(df_count_ingredients)] = new_entry

    df_ingredients_sorted = df_count_ingredients.sort_values(by='count', ascending=False)

    print("extracting most important ingredients")
    # safe 10 most important ingedrients in extra columns
    for i, r in df_ingredients_sorted.head(10).iterrows():
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

    #print("done with ingredients")
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

    # rename columns
    """ df_recipes.rename(columns={
        'CookTime': 'Recipe_CookTime',
        'PrepTime': 'Recipe_PrepTime',
        'Calories': 'Recipe_Calories',
        'FatContent': 'Recipe_FatContent',
        'SaturatedFatContent': 'Recipe_SaturatedFatContent',
        'CholesterolContent': 'Recipe_CholesterolContent',
        'SodiumContent': 'Recipe_SodiumContent',
        'CarbohydrateContent': 'Recipe_CarbohydrateContent',
        'FiberContent': 'Recipe_FiberContent',
        'SugarContent': 'Recipe_SugarContent',
        'ProteinContent': 'Recipe_ProteinContent',
        'RecipeServings': 'Recipe_RecipeServings',
        'RecipeYield': 'Recipe_RecipeYield',
    }, inplace=True)
    return df_recipes """


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
    df_recipes.rename(columns=lambda x: "recipes_" + x if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
    df_diet.rename(columns=lambda x: "diet_" + x if x not in ["AuthorId", "RecipeId"] else x, inplace=True)


def merge_df():
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
    
    return df_merged


def main():
    seed = 2024
    np.random.seed(seed)

    # clean diet
    print("start cleaning diet")
    clean_diet()
    print("done cleaning diet")

    # clean recipes
    print("start cleaning recipes")
    clean_recipes()
    print("done cleaning recipes")

    #clean request
    print("start cleaning requests")
    clean_requests()
    print("done cleaning requests")

    #rename columns
    print("start renaming columns")
    rename_columns()
    print("done renaming columns")
    
    #merge
    print("start merging")
    df_final = merge_df()
    print("done merging")
    
    #save as pickle
    df_final.to_pickle('cleaned_dataset.pkl')
    print("done saving as pickle")
    

    # check results
    df_final = pd.read_pickle('cleaned_dataset.pkl')
    print(df_final.head())
    print(df_final.info())


if __name__ == "__main__":
    main()
