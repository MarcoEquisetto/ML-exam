import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define random state var (the same one will be used in any method that requires it)
randomState = 42

def drawCorrelationMatrix(dataset):
    plt.figure(figsize=(12, 10))
    CM = dataset.corr(numeric_only=True).abs().style.background_gradient(axis=None, cmap='Reds')
    sns.heatmap(CM.data, annot=True, fmt=".2f", cmap='Reds')
    plt.title('Feature Correlation Matrix')
    plt.show()
    return CM

def plotQualityCheckGraph(scores, bestHyperparameter, bestAccuracy):
    # Create figure to show accuracy evolution graph and confusion matrix for best KNN 
    # iteration side by side
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, plot 1
    ax1.plot(range(1, len(scores)+1), scores, marker='o', markersize=4, color='steelblue')
    ax1.set_title(f'F1-Score vs. HyperParameter (Best HP = {bestHyperparameter})', fontsize=14)
    ax1.set_xlabel('HyperParameter value')
    ax1.set_ylabel('F1-Score')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(bestHyperparameter, color='red', linestyle='--', linewidth=2, label=f'Best HyperParameter = {bestHyperparameter}')
    ax1.scatter(bestHyperparameter, scores[bestHyperparameter-1], color='red', s=100, zorder=5)
    ax1.legend()
    ax1.text(
        0.02, 
        0.98, 
        f'Best F1-Score: {bestAccuracy:.4f}', 
        transform=ax1.transAxes, 
        fontsize=12,
        verticalalignment='top', 
        bbox=dict(boxstyle="round", facecolor="wheat")
    )

    # Subplot 2: Confusion Matrix of best KNN
    cm = confusion_matrix(y_val, predVal)
    ax2 = fig.add_subplot(1, 2, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['GALAXY', 'STAR', 'QSO'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title('Confusion Matrix (Validation Set)')
    plt.tight_layout()
    plt.show()


# Import the Star Classification Dataset and print its shape
dataset = pd.read_csv('../dataset/star_classification.csv')
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
X_rawTrain, X_rawTest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)



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
print(f"\n\n> Model Training and Evaluation\n> Models to be implemented:\n1. KNeighborsClassifier\n2. RandomForestClassifier\n3. Support Vector Machine (SVM)")
X_partial_train, X_val, y_partial_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=randomState)

# KNN training
Ks = range(1, 51)
KNNcrossValidationScores = []
print(f"\n\n> KNN training: evaluate KNearestNeighbors classifiers with K ranging from 1 to {max(Ks)} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for n_neighbors in Ks:
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(KNN, X_train, y_train, cv=5, scoring='f1_macro')
    KNNcrossValidationScores.append(scores.mean())
    print(f"KNN with k = {n_neighbors} has CV score = {scores.mean()}")

# Evaluate which was the K that fit best
bestK = Ks[np.argmax(KNNcrossValidationScores)]

# Retrain with that K to show graphs related to this iteration
print(f"> Best k value for KNN found to be {bestK} with F1-score = {max(KNNcrossValidationScores):.4f}... Retraining KNN with best k to show related graphs")
KNN = KNeighborsClassifier(bestK)
KNN.fit(X_train, y_train)
predVal = KNN.predict(X_val)

accuracy = sklearn.metrics.accuracy_score(y_val, predVal)
recall = sklearn.metrics.recall_score(y_val, predVal, average="macro")
precision = sklearn.metrics.precision_score(y_val, predVal, average="macro")
f1 = sklearn.metrics.f1_score(y_val, predVal, average="macro")

mse = mean_squared_error(y_val, predVal)
mae = mean_absolute_error(y_val, predVal)
r2 = r2_score(y_val, predVal)

plotQualityCheckGraph(KNNcrossValidationScores, bestK, max(KNNcrossValidationScores))
print(f"\n--- Classification Metrics ---")
print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f},\nRecall: {recall:.4f}\nF1 Score:  {f1:.4f}")

print(f"\n--- Regression Metrics ---")
print(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR2 Score:  {r2:.4f}")
print(f"\n> KNN evaluation completed. Best k is shown to be {bestK}")


# Random Forest Classifier training
estimators = range(1, 51)
RFcrossValidationScores = []
print(f"\n\n> Random Forest Classifier training: evaluate Random Forest classifiers with n_estimators ranging from 1 to {max(estimators)} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for n_estimators in estimators:
    RF = RandomForestClassifier(n_estimators=n_estimators, random_state=randomState)
    scores = cross_val_score(RF, X_train, y_train, cv=5, scoring='f1_macro')
    RFcrossValidationScores.append(scores.mean())
    print(f"RFC with n_estimators = {n_estimators} has CV score = {scores.mean()}")


bestEstimator = estimators[np.argmax(RFcrossValidationScores)]

print(f"> Best validation F1-score: {max(RFcrossValidationScores):.4f} achieved at n_estimators = {bestEstimator}... Retrain Random Forest with best n_estimators to show related graphs")
RF = RandomForestClassifier(n_estimators=bestEstimator, random_state=randomState)
RF.fit(X_train, y_train)
predVal = RF.predict(X_val)

accuracy = sklearn.metrics.accuracy_score(y_val, predVal)
recall = sklearn.metrics.recall_score(y_val, predVal, average="macro")
precision = sklearn.metrics.precision_score(y_val, predVal, average="macro")
f1 = sklearn.metrics.f1_score(y_val, predVal, average="macro")

mse = mean_squared_error(y_val, predVal)
mae = mean_absolute_error(y_val, predVal)
r2 = r2_score(y_val, predVal)

plotQualityCheckGraph(RFcrossValidationScores, bestEstimator, max(RFcrossValidationScores))
print(f"\n--- Classification Metrics ---")
print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f},\nRecall: {recall:.4f}\nF1 Score:  {f1:.4f}")

print(f"\n--- Regression Metrics ---")
print(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR2 Score:  {r2:.4f}")
print(f"\n> RFC evaluation completed")
