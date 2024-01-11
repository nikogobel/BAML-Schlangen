import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

#set the seed
seed = 2024
np.random.seed(seed)

# load the data
file_path_requests = "requests.csv"
file_path_reviews = "reviews.csv"
file_path_recipes = "recipes.csv"
file_path_diet = "diet.csv"

df_requests = pd.read_csv(file_path_requests)
df_reviews = pd.read_csv(file_path_reviews)
df_recipes = pd.read_csv(file_path_recipes)
df_diet = pd.read_csv(file_path_diet)

df_requests.rename(columns=lambda x: x + "_requests" if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
df_reviews.rename(columns=lambda x: x + "_reviews" if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
df_recipes.rename(columns=lambda x: x + "_recipes" if x not in ["AuthorId", "RecipeId"] else x, inplace=True)
df_diet.rename(columns=lambda x: x + "_diet" if x not in ["AuthorId", "RecipeId"] else x, inplace=True) 

#convert the data types to category
df_requests["HighCalories_requests"] = df_requests["HighCalories_requests"].astype(int)
df_requests["HighCalories_requests"] = df_requests["HighCalories_requests"].astype("category")

df_requests["HighProtein_requests"] = df_requests["HighProtein_requests"].astype("category")

df_requests["LowFat_requests"] = df_requests["LowFat_requests"].astype("category")

df_requests["LowSugar_requests"] = df_requests["LowSugar_requests"].astype("category")

df_requests["HighFiber_requests"] = df_requests["HighFiber_requests"].astype("category")



#join the dataframes
df_1= pd.merge(df_diet,df_reviews, on="AuthorId", how="inner")
df_2= pd.merge(df_1,df_recipes, on="RecipeId", how="inner")
df_final = pd.merge(df_2, df_requests, on=["AuthorId", "RecipeId"], how="inner")

print(df_final.head())
print(df_final.info())