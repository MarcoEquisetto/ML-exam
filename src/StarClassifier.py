# Import common libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Import evaluation tools
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline

# Custom helper functions
from helperFuncs import drawCorrelationMatrix, plotKernelPerformanceComparison, plotQualityCheckGraph, printClusterMetrics, plotLogQualityCheckGraph, displayConfusionMatrix, outlierDetection


randomState = 42

# Hyperparameters max ranges
maxK = 100
maxEstimators = 100

# Import the Dataset and print its shape
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
new = dataset.columns.difference(toMove).to_list() + toMove
dataset = dataset[new]


# Handle missing values
print(f"\n\n> Check for N/A values since they make the dataset more sparse and therefore hinder the accuracy of the models")
print(f'Missing values: {dataset.isna().any().any()}')
print(f'Duplicated rows: {dataset.duplicated().any()}')
if not (dataset.isna().any().any() | dataset.duplicated().any()):
    print(f"> No N/A values or duplicated rows found in the dataset")


# Pre-check for error values
print(f"\n\n> Check for error values in the dataset")
print(dataset.describe())
print(f"\n> Deleting Errors...")
dataset = dataset[dataset['u'] != -9999]
dataset = dataset[dataset['g'] != -9999]
dataset = dataset[dataset['z'] != -9999]
print(f"> Error values removed... New Shape: {dataset.shape}")



## Outlier Detection
detectionFeatures = ['u', 'g', 'r', 'i', 'z', 'redshift']
X_tmp = dataset[detectionFeatures]

# Run Detection
mask, scores, threshold = outlierDetection(X_tmp)

# Visualize Detection Results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.suptitle("GMM Outlier Detection Results", fontsize=16)

# 1. Histogram of Likelihoods
sns.histplot(scores, bins=50, kde=True, ax=axes[0], color='skyblue')
axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
axes[0].set_title("Log-Likelihood Distribution")
axes[0].set_xlabel("Log-Likelihood (Density)")
axes[0].legend()

# 2. PCA Projection
pca_vis = PCA(n_components=2, random_state=randomState)
X_pca_vis = pca_vis.fit_transform(StandardScaler().fit_transform(X_tmp))
sns.scatterplot(x=X_pca_vis[:,0], y=X_pca_vis[:,1], hue=mask, palette={False: 'gray', True: 'red'}, alpha=0.6, ax=axes[1])
axes[1].set_title(f"Detected Outliers (Red: {np.sum(mask)} points)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

plt.tight_layout()
plt.show()

# Apply Removal
print(f"> Removing {np.sum(mask)} outliers detected via GMM...")
dataset = dataset[~mask]
dataset.reset_index(drop=True, inplace=True)
print(f"> Outliers removed. Final Shape: {dataset.shape}")



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

# Subtracting magnitudes gives the color
dataset['u_g'] = dataset['u'] - dataset['g']
dataset['g_r'] = dataset['g'] - dataset['r']
dataset['r_i'] = dataset['r'] - dataset['i']
dataset['i_z'] = dataset['i'] - dataset['z']
dataset = dataset.drop(columns = ['u', 'g', 'i', 'z', 'redshift', 'alpha', 'delta'])
for pair in pairs:
    toDrop = pair[1]
    if toDrop in dataset.columns:
        dataset = dataset.drop(columns = [toDrop])
        print(f"> Dropped feature: {toDrop}")
print(f"\n\n> Optimized correlation matrix:\n{drawCorrelationMatrix(dataset).data}")


# Split the dataset into features and target variable
print(f"\n\n> Split dataset into features and target variable")
targetName = "class"
X = dataset.drop(columns = [targetName])
y = dataset[targetName]
print(f"Features used for classification: {X.columns.to_list()}")
print(f"Shape of features: {X.shape}")


## Split the dataset into training and testing sets (80% train, 20% test)
print(f"\n\n> Split the dataset into Training and Testing sets (80%, 20%)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = randomState)



## Model Training and Evaluation
print(f"\n\n> Model Training and Evaluation\n> Models to be implemented:\n1. KNeighborsClassifier\n2. RandomForestClassifier\n3. Support Vector Machine (SVM)")

# KNN training
Ks = range(1, maxK + 1)
KNNcrossValidationScores = []
print(f"\n\n> KNN training: evaluate KNNs with k ranging from 1 to {max(Ks)} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for n_neighbors in Ks:
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = n_neighbors))
    scores = cross_val_score(pipeline, X_train, y_train, cv = 5, scoring = 'f1_macro')
    KNNcrossValidationScores.append(scores.mean())
    print(f"> n_neighbors = {n_neighbors}, F1-score (mean of batch) = {scores.mean():.4f}")

# Extrapolate best K
bestK = Ks[np.argmax(KNNcrossValidationScores)]

# TODO: DELETE THIS
#bestK = 15

# Retrain with that K to show graphs related to this iteration
print(f"\n> Best k value for KNN found to be {bestK} with F1-score = {max(KNNcrossValidationScores):.4f}: Retraining KNN with best k to show related graphs and metrics")
bestKNNPipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = bestK))
bestKNNPipeline.fit(X_train, y_train)
predVal = bestKNNPipeline.predict(X_test)

plotQualityCheckGraph(KNNcrossValidationScores, bestK, max(KNNcrossValidationScores), "KNearestNeighbors")
displayConfusionMatrix(y_test, predVal, f"Best KNN (HP = {bestK})")

print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")



# Random Forest Classifier training
estimators = range(1, maxEstimators + 1)
RFcrossValidationScores = []
print(f"\n\n> Random Forest Classifier training: evaluate RFCs with n_estimators ranging from 1 to {max(estimators)} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for n_estimators in estimators:
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = randomState))
    scores = cross_val_score(pipeline, X_train, y_train, cv = 5, scoring = 'f1_macro')
    RFcrossValidationScores.append(scores.mean())
    print(f"> n_estimators = {n_estimators}, F1-score (mean of batch) = {scores.mean():.4f}")

# Extrapolate best n_estimators
bestEstimator = estimators[np.argmax(RFcrossValidationScores)]

# TODO: DELETE THIS 
#bestEstimator = 96

print(f"\n> Best validation F1-score: {max(RFcrossValidationScores):.4f} achieved at n_estimators = {bestEstimator}... Retrain RFC with best n_estimators to show related graphs")
bestRFCPipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = bestEstimator, random_state = randomState))
bestRFCPipeline.fit(X_train, y_train)
predVal = bestRFCPipeline.predict(X_test)

plotQualityCheckGraph(RFcrossValidationScores, bestEstimator, max(RFcrossValidationScores), "RandomForestClassifier")
displayConfusionMatrix(y_test, predVal, f"Best Random Forest (HP = {bestEstimator})")

print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")




# SVM training
sampleIndex = np.random.choice(X_train.index, size=int(len(X_train) * 0.1), replace = False)
Xsplit = X_train.loc[sampleIndex]
ysplit = y_train.loc[sampleIndex]
print(f"\n> Tuning SVM on subset of {len(Xsplit)} rows...")
 
Cs = [0.001, 0.01, 0.1, 1, 10, 100]
kernels = ['rbf', 'linear', 'sigmoid']
kernelResults = {}
bestScore = -1
bestParams = {}
print(f"\n\n> SVM training: evaluate SVM classifiers with C ranging inside {Cs} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for kernel in kernels:   
    scores = []
    print(f"\n> Evaluating: {kernel.upper()}")
    
    for C in Cs:
        SVCPipeline = make_pipeline(StandardScaler(), SVC(C = C, kernel = kernel, random_state = randomState))
        cv_scores = cross_val_score(SVCPipeline, Xsplit, ysplit, cv = 5, scoring = 'f1_macro')
        mean_score = cv_scores.mean()
        scores.append(mean_score)
        
        print(f"  > C = {C:<5} | F1-score = {mean_score:.4f}")
        if mean_score > bestScore:
            bestScore = mean_score
            bestParams = {'C': C, 'kernel': kernel}
    
    kernelResults[kernel] = scores

# Extrapolate best C
print(f"\n> Generating Kernel Comparison Graph including Polynomial Degrees")
plotKernelPerformanceComparison(kernelResults, Cs)

# Retrain best model on full data
bestC = bestParams['C']
bestKernel = bestParams['kernel']

print(f"\n> Best Validation Score: {bestScore:.4f}")
print(f"> Best Params: Kernel='{bestKernel}', C={bestC}: Retraining model with best params to show related graphs and metrics")

bestSVCPipeline = make_pipeline(StandardScaler(), SVC(C = bestC, kernel = bestKernel, random_state = randomState))
bestSVCPipeline.fit(X_train, y_train)
predVal = bestSVCPipeline.predict(X_test)

displayConfusionMatrix(y_test, predVal, f"Best SVM (HP = C={bestC}, kernel={bestKernel})")

# Final metrics
print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")



# Logistic Regression training
LR_Cs = [0.001, 0.01, 0.1, 1, 10, 100]
LRcrossValidationScores = []

print(f"\n\n> Logistic Regression training: evaluate LRs with C ranging inside {LR_Cs} using 5-Fold Cross Validation and F1-score (macro) as metrics")
for C in LR_Cs:
    LRPipeline = make_pipeline(StandardScaler(), LogisticRegression(C = C, random_state = randomState))
    scores = cross_val_score(LRPipeline, X_train, y_train, cv = 5, scoring = 'f1_macro')
    LRcrossValidationScores.append(scores.mean())
    print(f"> C = {C}, F1-score (mean of batch) = {scores.mean():.4f}")

# Extrapolate best C
bestLR_C = LR_Cs[np.argmax(LRcrossValidationScores)]

print(f"\n> Best validation F1-score: {max(LRcrossValidationScores):.4f} achieved at C = {bestLR_C}... Retrain Logistic Regression with best C to show related graphs")

bestLRPipeline = make_pipeline(StandardScaler(), LogisticRegression(C = bestLR_C, random_state = randomState))
bestLRPipeline.fit(X_train, y_train)
predVal = bestLRPipeline.predict(X_test)
plotLogQualityCheckGraph(LRcrossValidationScores, LR_Cs, bestLR_C, max(LRcrossValidationScores), "Logistic Regression")

print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")



## Clustering
print(f"\n\n> Clustering models to be implemented:\n1. KMeans\n2. GMM\nBoth models will be trained on \"raw\" data and on PCA-optimized data to compare results")

# Traing new scaler for clustering
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
PCA = PCA(n_components = 2, random_state = randomState)
X_PCA = PCA.fit_transform(X_scaled)


# KMeans Clustering
KmeansRAW = KMeans(n_clusters = 3, random_state = randomState)
raw_Kmeans = KmeansRAW.fit_predict(X_scaled)
KmeansPCA = KMeans(n_clusters = 3, random_state = randomState)
PCA_Kmeans = KmeansPCA.fit_predict(X_PCA)


# GMM Clustering
GMMRAW = GaussianMixture(n_components = 3, random_state = randomState)
raw_GMM = GMMRAW.fit_predict(X_scaled)
GMPPCA = GaussianMixture(n_components = 3, random_state = randomState)
PCA_GMM = GMPPCA.fit_predict(X_PCA)


fig, axes = plt.subplots(2, 2, figsize = (18, 12))
# KMeans Plots
sns.scatterplot(x = X_PCA[:, 0], y = X_PCA[:, 1], hue = raw_Kmeans, palette = 'viridis', alpha = 0.6, ax = axes[0, 0])
axes[0, 0].set_title('KMeans (Trained on Raw Data)')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
sns.scatterplot(x = X_PCA[:, 0], y = X_PCA[:, 1], hue = PCA_Kmeans, palette = 'viridis', alpha = 0.6, ax = axes[0, 1])
axes[0, 1].set_title('KMeans (Trained on PCA Data)')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')

# GMM Plots
sns.scatterplot(x = X_PCA[:, 0], y = X_PCA[:, 1], hue = raw_GMM, palette = 'magma', alpha = 0.6, ax = axes[1, 0])
axes[1, 0].set_title('GMM (Trained on Raw Data)')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
sns.scatterplot(x = X_PCA[:, 0], y = X_PCA[:, 1], hue = PCA_GMM, palette = 'magma', alpha = 0.6, ax = axes[1, 1])
axes[1, 1].set_title('GMM (Trained on PCA Data)')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')

plt.tight_layout()
plt.show()

print("\n\n> Clustering Quantitative Evaluation")
printClusterMetrics("KMeans (Raw Data)", y, raw_Kmeans, X_scaled)
printClusterMetrics("KMeans (PCA Data)", y, PCA_Kmeans, X_PCA)
printClusterMetrics("GMM (Raw Data)", y, raw_GMM, X_scaled)
printClusterMetrics("GMM (PCA Data)", y, PCA_GMM, X_PCA)