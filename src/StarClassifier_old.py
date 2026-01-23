import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

randomState = 42

# TODO: remove these declarations
maxK = 7
maxEstimators = 5
maxC = 6
maxClusters = 10

def drawCorrelationMatrix(dataset):
    plt.figure(figsize=(12, 10))
    CM = dataset.corr(numeric_only=True).abs().style.background_gradient(axis=None, cmap='Reds')
    sns.heatmap(CM.data, annot=True, fmt=".2f", cmap='Reds')
    plt.title('Feature Correlation Matrix')
    plt.show()
    return CM

def plotQualityCheckGraph(scores, bestHyperparameter, bestScore, model_name):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(scores)+1), scores, marker='o', markersize=4, color='steelblue')
    plt.title(f'{model_name}: F1-Score vs. HyperParameter (Best = {bestHyperparameter})', fontsize=14)
    plt.xlabel('HyperParameter Value')
    plt.ylabel('F1-Score (macro)')
    plt.grid(True, alpha=0.3)
    plt.axvline(bestHyperparameter, color='red', linestyle='--', linewidth=2, label=f'Best HP = {bestHyperparameter}')
    plt.scatter(bestHyperparameter, bestScore, color='red', s=100, zorder=5)
    plt.legend()
    plt.show()


# Import the Star Classification Dataset and print its shape
dataset = pd.read_csv('./dataset/star_classification.csv')
print("\n\n> Star Classification Dataset")
print(dataset)
print(f"Shape: {dataset.shape}")


## Data Cleansing
# Drop ID columns that aren't useful for classification and move class column to the end (just for convenience)
columnsToDrop = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'MJD', 'plate', 'fiber_ID']
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
datasetMerge = dataset.melt(id_vars=['class'], var_name='variable', value_name='value')
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

print(f"\n> Drop highly correlated features, create synthetic ones and recompute correlation matrix")
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)


## Preprocessing 
print(f"\n\n> Preprocessing\n> Given the logarithmic nature of the redshift feature, apply log1p transformation and scaling to it")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)



## Model Training and Evaluation
print(f"\n\n> Model Training and Evaluation\n> Models to be implemented:\n1. KNeighborsClassifier\n2. RandomForestClassifier\n3. Support Vector Machine (SVM)")
X_partial_train, X_val, y_partial_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=randomState)



# KNN training
Ks = range(1, maxK)
KNNcrossValidationScores = []
print(f"\n\n> KNN training: evaluate KNearestNeighbors classifiers with K ranging from 1 to {max(Ks)} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for n_neighbors in Ks:
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(KNN, X_train, y_train, cv=5, scoring='f1_macro')
    KNNcrossValidationScores.append(scores.mean())
    print(f"> n_neighbors = {n_neighbors}, F1-score (mean of batch) = {scores.mean():.4f}")

# Evaluate which was the K that fit best
bestK = Ks[np.argmax(KNNcrossValidationScores)]

# Retrain with that K to show graphs related to this iteration
print(f"\n> Best k value for KNN found to be {bestK} with F1-score = {max(KNNcrossValidationScores):.4f}: Retraining KNN with best k to show related graphs")
bestKNN = KNeighborsClassifier(bestK)
bestKNN.fit(X_train, y_train)
predVal = bestKNN.predict(X_test)

plotQualityCheckGraph(KNNcrossValidationScores, bestK, max(KNNcrossValidationScores), "KNN")
print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")



# Random Forest Classifier training
estimators = range(1, maxEstimators)
RFcrossValidationScores = []
print(f"\n\n> Random Forest Classifier training: evaluate Random Forest classifiers with n_estimators ranging from 1 to {max(estimators)} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for n_estimators in estimators:
    RF = RandomForestClassifier(n_estimators=n_estimators, random_state=randomState)
    scores = cross_val_score(RF, X_train, y_train, cv=5, scoring='f1_macro')
    RFcrossValidationScores.append(scores.mean())
    print(f"> n_estimators = {n_estimators}, F1-score (mean of batch) = {scores.mean():.4f}")

bestEstimator = estimators[np.argmax(RFcrossValidationScores)]

print(f"\n> Best validation F1-score: {max(RFcrossValidationScores):.4f} achieved at n_estimators = {bestEstimator}... Retrain Random Forest with best n_estimators to show related graphs")
RF = RandomForestClassifier(n_estimators=bestEstimator, random_state=randomState)
RF.fit(X_train, y_train)
predVal = RF.predict(X_test)

plotQualityCheckGraph(RFcrossValidationScores, bestEstimator, max(RFcrossValidationScores), "RFC")
print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")



## Clustering
print(f"\n\n> Clustering\n> Models to be implemented:\n1. KMeans\n2. GMM with PCA")
# KMeans Clustering
print(f"\n\n> KMeans Clustering")
KMeansModel = KMeans(n_clusters=3, random_state=randomState)
KMeanslabels = KMeansModel.fit_predict(X)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=KMeanslabels, palette='viridis', alpha=0.6)
plt.title('KMeans Clustering on Original Data')


## GMM optimized with PCA for dimensionality reduction
print(f"\n\n> GMM optimized with PCA for dimensionality reduction")
PCA = PCA(n_components=2, random_state=randomState)
X_reduced = PCA.fit_transform(X)
print(f"> Explained variance ratio by the 2 principal components: {PCA.explained_variance_ratio_}")
loadings = pd.DataFrame(
    PCA.components_.T, 
    columns=['PC1', 'PC2'], 
    index=X.columns
)
print("\n> Feature importance (PCA Loadings):\n", loadings)

GMM_PCA = GaussianMixture(n_components=3, random_state=randomState)
GMMlabels = GMM_PCA.fit_predict(X_reduced)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=GMMlabels, palette='viridis', alpha=0.6)
plt.title('GMM Clustering on PCA reduced Data')
plt.xlabel('Component1')
plt.ylabel('Component2')
plt.show()

