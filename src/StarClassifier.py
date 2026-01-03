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


# Pre-check for outliers
print(f"\n\n> Check for outliers in the dataset")
print(dataset.describe())
print(f"\n> Deleting outliers...")
dataset = dataset[dataset['u'] != -9999]
dataset = dataset[dataset['g'] != -9999]
dataset = dataset[dataset['z'] != -9999]
print(f"\n> Outliers removed. New Shape: {dataset.shape}\n> Check for outliers after deletion")
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
LE = LabelEncoder()
dataset['class'] = LE.fit_transform(dataset['class'])
print(dataset.head())

def drawCorrelationMatrix(dataset):
    plt.figure(figsize=(12, 10))
    CM = dataset.corr(numeric_only=True).abs().style.background_gradient(axis=None, cmap='Reds')
    sns.heatmap(CM.data, annot=True, fmt=".2f", cmap='Reds')
    plt.title('Feature Correlation Matrix')
    plt.show()
    return CM

# Feature correlation analysis
print(f"\n\n> Feature correlation analysis")
CM = drawCorrelationMatrix(dataset)
print(f"\n\n> Arbitrarily considering features with correlation index higher than 0.9 as highly correlated, the following pairs are highly correlated:")
pairs = []
for i in range(len(CM.data.columns)):
    for j in range(i):
        if CM.data.iloc[i, j] > 0.9 and i != j:
            pair = (CM.data.columns[i], CM.data.columns[j], CM.data.iloc[i, j])
            pairs.append(pair)
            print(f"> {pair[0]} and {pair[1]} with correlation index {pair[2]:.2f}")

print(f"\n> Drop highly correlated features and recompute correlation matrix")
for pair in pairs:
    # Keep the first out of the pair of features
    toDrop = pair[1]
    if toDrop in dataset.columns:
        dataset = dataset.drop(columns=[toDrop])
        print(f"> Dropped feature: {toDrop}")

print(f"\n\n> Optimized correlation matrix:\n{drawCorrelationMatrix(dataset).data}")


# Split the dataset into features and target variable
print(f"\n\n> Split dataset into features and target variable")
targetName = "class"
X = dataset.drop(columns=[targetName])
y = dataset[targetName]