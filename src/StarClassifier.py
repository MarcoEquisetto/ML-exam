import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sklearn.metrics


# Import the Star Classification Dataset and print its shape
dataset = pd.read_csv('./dataset/star_classification.csv')
print("\n\n> Star Classification Dataset")
print(dataset)
print(f"Shape: {dataset.shape}")


## Data Cleansing
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
print(f"> No N/A values or duplicated rows found in the dataset")


# Pre-check for outliers
print(f"\n\n> Check for outliers in the dataset")
print(dataset.describe())
print(f"\n> Deleting outliers...")
dataset = dataset[dataset['u'] != -9999]
dataset = dataset[dataset['g'] != -9999]
dataset = dataset[dataset['z'] != -9999]
print(f"> Outliers removed. New Shape: {dataset.shape}")


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
print(f"Features used for classification: {X.columns.to_list()}")
print(f"Shape of features: {X.shape}")



## Split the dataset into training and testing sets (80% train, 20% test)
print(f"\n\n> Split the dataset into Training and Testing sets (80%, 20%)")
X_rawTrain, X_rawTest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## Preprocessing 
print(f"\n\n> Preprocessing\n> Given the logarithmic nature of the redshift feature, apply log1p transformation and scaling to it")
X_train = X_rawTrain.copy()
X_test = X_rawTest.copy()
X_test.redshift = np.log1p(X_test.redshift)
X_train.redshift = np.log1p(X_train.redshift)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)



## Model Training and Evaluation
print(f"\n\n> Model Training and Evaluation\n> Models to be implemented:\nKNN\nLR\nRANDOM FOREST\nSVM")
KNN = KNeighborsClassifier(),
LR = LinearRegression(),
RF = RandomForestClassifier(),
SVM = svm.SVC()

# KNN training
accuracies = []
max_k = 50
X_partial_train, X_val, y_partial_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\n\n> KNN training: we will evaluate KNearestNeighbors classifiers with K ranging from 1 to {max_k}")
for n_neighbors in range(1, max_k + 1):
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit model to data
    # the model will be trained using the training data X_train and the labels y_train.
    KNN.fit(X_train, y_train)


    # VALIDATION
    # "KNN" represents the K-Nearest Neighbors model that we trained earlier.
    # predict() is a method to make predictions using the trained model.
    # The array pred_val will contain the model's predictions for the validation data (X_val).
    pred_val = KNN.predict(X_val)


    # CALCULATE ACCURACY
    # "pred_val == y_val": This expression creates a boolean array where each element
    # is True if the prediction matches the true label for the corresponding observation in the validation data, otherwise it is False.
    # "np.mean(...)" calculates the fraction of "True" elements in a boolean array, thus representing the percentage of correct predictions out of the total predictions.
    # y_val: Ground truth for validation data. Represents the true labels in the validation data.
    accuracy = np.mean(pred_val == y_val)
    accuracies.append(accuracy)
    if n_neighbors % 10 == 0:
        print(f"   k={n_neighbors}, Accuracy={accuracy:.4f}")

# Evaluate which was the K that fit best
best_k = np.argmax(accuracies) + 1
best_accuracy = max(accuracies)

# Retrain with that K to show graphs related to this iteration
KNN = KNeighborsClassifier(best_k)
KNN.fit(X_train, y_train)
pred_val = KNN.predict(X_val)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create figure to show accuracy evolution graph and confusion matrix for best KNN 
# iteration side by side
fig = plt.figure(figsize=(16, 6))
print(f"Best validation accuracy: {best_accuracy:.4f} achieved at k = {best_k}")
ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, plot 1
ax1.plot(range(1, len(accuracies)+1), accuracies, marker='o', markersize=4, color='steelblue')
ax1.set_title(f'Validation Accuracy vs. k (Best k = {best_k})', fontsize=14)
ax1.set_xlabel('Number of Neighbors (k)')
ax1.set_ylabel('Validation Accuracy')
ax1.grid(True, alpha=0.3)
ax1.axvline(best_k, color='red', linestyle='--', linewidth=2, label=f'Best k = {best_k}')
ax1.scatter(best_k, accuracies[best_k-1], color='red', s=100, zorder=5)
ax1.legend()
ax1.text(
    0.02, 
    0.98, 
    f'Best accuracy: {best_accuracy:.4f}', 
    transform=ax1.transAxes, 
    fontsize=12, 
    verticalalignment='top', 
    bbox=dict(boxstyle="round", facecolor="wheat")
)

# Subplot 2: Confusion Matrix
cm = confusion_matrix(y_val, pred_val)
ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, plot 2
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1', 'Class 2'])
disp.plot(ax=ax2, cmap='Blues', values_format='d')
ax2.set_title('Confusion Matrix (Validation Set)')

# Calculating the Confusion Matrix
# Calculate the confusion matrix based on the real labels "y_val" and the predicted ones by the KNN model "pred_val"
print("\n\nHere is the confusion matrix for the best-fit KNN classifier")
print("The total amount of data predicted as 0:", sum(pred_val==0))
print("The total amount of data that is 0 in reality:", sum(y_val==0))
print("The total amount of data predicted as 0:", sum(pred_val==1))
print("The total amount of data that is 0 in reality:", sum(y_val==1))
print("The total amount of data predicted as 0:", sum(pred_val==2))
print("The total amount of data that is 0 in reality:", sum(y_val==2))

plt.tight_layout()
plt.show()

# Calculate accuracy, recall, and precision with scikit-learn
accuracy = sklearn.metrics.accuracy_score(y_val, pred_val)
recall = sklearn.metrics.recall_score(y_val, pred_val, average="macro")
precision = sklearn.metrics.precision_score(y_val, pred_val, average="macro")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")