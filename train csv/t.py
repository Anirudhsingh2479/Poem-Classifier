import pandas as pd

df = pd.read_csv("Cleaned_PoetryData.csv")
unique_genres = df['Genre'].unique()
print(unique_genres)