import pandas as pd
import numpy as np

# Import the Star Classification Dataset and print its shape
dataset = pd.read_csv('./dataset/star_classification.csv')
print("\n\n> Star Classification Dataset")
print(dataset)
print(f"Shape: {dataset.shape}")

## Data Preprocessing
# Handle missing values
print(f"\n\n> Check for N/A values since they make the dataset more sparse and therefore hinder the accuracy of the models")
print(f'Missing values: {dataset.isna().any().any()}')
print(f'Duplicated rows: {dataset.duplicated().any()}')
print(f"\n\n> No N/A values or duplicated rows found in the dataset")


# Check for outliars
print(f"\n\n> Check for outliars in the dataset")
print(dataset.describe())
print(f"\n> Deleting outliars...")
dataset.drop(
    index = dataset[
        (dataset.u==dataset.u.min()) | 
        (dataset.g==dataset.g.min()) | 
        (dataset.z==dataset.z.min())
    ].index[0], 
    inplace=True
)
print(f'\n> Check for outliars after deletion')
print(dataset.describe())
