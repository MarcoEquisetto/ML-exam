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
# Drop ID columns that aren't useful for classification and move class column to the end (just for convenience)
columnsToDrop = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID']
toMove = ['class']
dataset = dataset.drop(columns=columnsToDrop)
print(f"\n> Dropping ID columns from dataset...\n> New Shape: {dataset.shape}")
new = dataset.columns.difference(toMove).to_list()+toMove
dataset = dataset[new]


# Handle missing values
print(f"\n\n> Check for N/A values since they make the dataset more sparse and therefore hinder the accuracy of the models")
print(f'Missing values: {dataset.isna().any().any()}')
print(f'Duplicated rows: {dataset.duplicated().any()}')
print(f"\n\n> No N/A values or duplicated rows found in the dataset")


# Check for 'dirty' data (not outliers yet)
print(f"\n\n> Check for outliars in the dataset")
print(dataset.describe())
print(f"\n> Deleting outliars...")
dataset = dataset[dataset['u'] != -9999]
dataset = dataset[dataset['g'] != -9999]
dataset = dataset[dataset['z'] != -9999]
print(f"\n> Outliers removed. New Shape: {dataset.shape}\n> Check for outliars after deletion")
print(dataset.describe())


# Check class distribution
print(f"\n\n> Class distribution (percentage):\n{dataset['class'].value_counts(normalize=True).mul(100).round(2)}")
datasetToPlot = dataset.drop(columns=['MJD', 'plate', 'fiber_ID'])
datasetMerge = datasetToPlot.melt(id_vars=['class'], var_name='variable', value_name='value')
sns.displot(
    data=datasetMerge,
    x='value', 
    col='variable',
    hue='class',
    kind='hist',
    height=4,
    aspect=2,
    kde=True,
    legend=True,
    col_wrap=2,
    facet_kws={'sharex': False, 'sharey': False},
    common_bins=False
)
plt.show()


# Encoding of categorical features
# 'class', the target feature, is categorical (GALAXY, STAR, QSO), but it is the only one
print(f"\n\n> Encoding categorical features...")
label_encoder = LabelEncoder()
dataset['class'] = label_encoder.fit_transform(dataset['class'])
print(dataset.head())


# Feature correlation analysis
print(f"\n\n> Feature correlation analysis")
plt.figure(figsize=(12, 10))
correlation_matrix = dataset.corr(numeric_only=True).abs().style.background_gradient(axis=None, cmap='Reds')
sns.heatmap(correlation_matrix.data, annot=True, fmt=".2f", cmap='Reds')
plt.title('Feature Correlation Matrix')
plt.show()


# Outlier detection



# Split the dataset into features and target variable
print(f"\n\n> Split dataset into features and target variable")
target_name = "class"
X = dataset.drop(columns=[target_name])
Y = dataset[target_name]
