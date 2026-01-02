import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Import the Star Classification Dataset and print its shape
dataset = pd.read_csv('./dataset/star_classification.csv')
print("\n\n> Star Classification Dataset")
print(dataset)
print(f"Shape: {dataset.shape}")


## Data Preprocessing
# Drop ID columns that aren't useful for classification
columnsToDrop = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID']
dataset = dataset.drop(columns=columnsToDrop)
print(f"\n> Dropped ID columns from dataset.\n> New Shape: {dataset.shape}")

# Handle missing values
print(f"\n\n> Check for N/A values since they make the dataset more sparse and therefore hinder the accuracy of the models")
print(f'Missing values: {dataset.isna().any().any()}')
print(f'Duplicated rows: {dataset.duplicated().any()}')
print(f"\n\n> No N/A values or duplicated rows found in the dataset")


# Check for outliars
print(f"\n\n> Check for outliars in the dataset")
print(dataset.describe())
print(f"\n> Deleting outliars...")
dataset = dataset[dataset['u'] != -9999]
dataset = dataset[dataset['g'] != -9999]
dataset = dataset[dataset['z'] != -9999]
print(f"\n> Outliers removed. New Shape: {dataset.shape}\n> Check for outliars after deletion")
print(dataset.describe())


# Encoding of categorical features
# 'class', the target feature, is categorical (GALAXY, STAR, QSO), but it is the only one
print(f"\n\n> Encoding categorical features using Label Encoding")
label_encoder = LabelEncoder()
dataset['class'] = label_encoder.fit_transform(dataset['class'])
print(f"Classes: {label_encoder.classes_}")



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


