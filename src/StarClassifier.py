import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the Star Classification Dataset and print its shape
dataset = pd.read_csv('./dataset/star_classification.csv')
print("\n\n> Star Classification Dataset")
print(dataset)
print(f"Shape: {dataset.shape}")


## Data Preprocessing
# Check data types (integer, object, float...) of the columns
target_name = "class"
X = dataset.iloc[:, dataset.columns != target_name]  # The features are stored in the other columns.
Y = dataset.iloc[:, dataset.columns == target_name]  # The labels are stored in the "class" column.
print(f"\n\n> Check data types of features and labels")
print(X.dtypes)
print("------------------------")
print(Y.dtypes)
print(f"\n> Since there are no caterogical features, no encoding is needed!")


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
        (dataset.u == dataset.u.min()) | 
        (dataset.g == dataset.g.min()) | 
        (dataset.z == dataset.z.min())
    ].index[0], 
    inplace=True
)
print(f'\n> Check for outliars after deletion')
print(dataset.describe())


# Check class distribution
print(f"\n\n> Check class distribution (percentage)")
print(Y.value_counts(normalize=True).mul(100).round(2))
print(f"\n\n> Data distribution visualization")
dataset.iloc[:, :-1].hist(bins=30, figsize=(15, 10))
plt.tight_layout()


# Check correlation between features
print(f"\n\n> Check correlation between features")
correlationMatrix = X.corr()
print(correlationMatrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlationMatrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()


