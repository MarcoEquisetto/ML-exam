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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import plot_tree

# Custom helper functions
from helperFuncs import drawCorrelationMatrix, plotKernelPerformanceComparison, plotQualityCheckGraph, printClusterMetrics, plotLogQualityCheckGraph, displayConfusionMatrix, outlierDetection


randomState = 42

# Hyperparameters max ranges
maxK = 100
maxEstimators = 150

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
plt.figure(figsize=(7, 7))
class_counts = dataset['class'].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Class Distribution Imbalance', fontsize=15)
plt.show()

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


# Feature correlation analysis
print(f"\n\n> Feature correlation analysis")
CM = drawCorrelationMatrix(dataset)

print(f"\n\n> Generating Model-Specific Feature Sets...")
# 1. Create Engineered Features
dataset['u_g'] = dataset['u'] - dataset['g']
dataset['g_r'] = dataset['g'] - dataset['r']
dataset['r_i'] = dataset['r'] - dataset['i']
dataset['i_z'] = dataset['i'] - dataset['z']

# Drop 'redshift' as per requirement, and 'alpha', 'delta' as they are positional
cols_to_exclude = ['redshift', 'alpha', 'delta']
dataset = dataset.drop(columns=[c for c in cols_to_exclude if c in dataset.columns])

print(f"\n> Optimized correlation matrix (All Features):\n{drawCorrelationMatrix(dataset).data}")


X_tree = dataset.drop(columns=['class'])
X_linear = dataset.drop(columns=['class', 'u', 'g', 'r', 'i', 'z'])
y = dataset['class']

print(f"\n> Feature Sets Prepared:")
print(f"  1. X_tree (for Random Forest): {X_tree.shape} - Features: {X_tree.columns.to_list()}")
print(f"  2. X_linear (for KNN/SVM/LR):  {X_linear.shape} - Features: {X_linear.columns.to_list()}")


## Split the dataset into training and testing sets (80% train, 20% test)
print(f"\n\n> Split the dataset into Training and Testing sets (80%, 20%)")

X_train_tree, X_test_tree, y_train, y_test = train_test_split(X_tree, y, test_size=0.2, random_state=randomState)
X_train_lin, X_test_lin, _, _ = train_test_split(X_linear, y, test_size=0.2, random_state=randomState)


## Model Training and Evaluation
print(f"\n\n> Model Training and Evaluation\n> Models to be implemented:\n1. KNeighborsClassifier\n2. RandomForestClassifier\n3. Support Vector Machine (SVM)\n4. Logistic Regression (with PCA)")

# KNN training
Ks = range(1, maxK + 1)
KNNcrossValidationScores = []
print(f"\n\n> KNN training: evaluate KNNs with k ranging from 1 to {max(Ks)}")
for n_neighbors in Ks:
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = n_neighbors))
    scores = cross_val_score(pipeline, X_train_lin, y_train, cv = 5, scoring = 'f1_macro')
    KNNcrossValidationScores.append(scores.mean())
    print(f"> n_neighbors = {n_neighbors}, F1-score (mean of batch) = {scores.mean():.4f}")

bestK = Ks[np.argmax(KNNcrossValidationScores)]
print(f"\n> Best k for KNN: {bestK}. Retraining...")

bestKNNPipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = bestK))
bestKNNPipeline.fit(X_train_lin, y_train)
predVal = bestKNNPipeline.predict(X_test_lin)

plotQualityCheckGraph(KNNcrossValidationScores, Ks, bestK, max(KNNcrossValidationScores), "KNearestNeighbors")
displayConfusionMatrix(y_test, predVal, f"Best KNN (HP = {bestK})")

print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")


# RFC training 
estimators = range(50, maxEstimators + 1)
print(f"\n\n> Random Forest training: evaluate RFCs with n_estimators ranging from 50 to {max(estimators)}")
RFcrossValidationScores = []
for n_estimators in estimators:
    pipeline = make_pipeline(RandomForestClassifier(n_estimators = n_estimators, random_state = randomState, class_weight = 'balanced', n_jobs=-1))
    scores = cross_val_score(pipeline, X_train_tree, y_train, cv = 5, scoring = 'f1_macro')
    RFcrossValidationScores.append(scores.mean())
    print(f"> n_estimators = {n_estimators}, F1-score (mean of batch) = {scores.mean():.4f}")

bestEstimator = estimators[np.argmax(RFcrossValidationScores)]
print(f"\n> Best n_estimators for RF: {bestEstimator}. Retraining...")

bestRFCPipeline = make_pipeline(RandomForestClassifier(n_estimators = bestEstimator, random_state = randomState, class_weight = 'balanced', n_jobs=-1))
bestRFCPipeline.fit(X_train_tree, y_train)
predVal = bestRFCPipeline.predict(X_test_tree)

plotQualityCheckGraph(RFcrossValidationScores, estimators, bestEstimator, max(RFcrossValidationScores), "RandomForestClassifier")
displayConfusionMatrix(y_test, predVal, f"Best Random Forest (HP = {bestEstimator})")
print("\n\n> Analyzing Feature Distribution in Random Forest...")
rf_model = bestRFCPipeline.named_steps['randomforestclassifier']
feature_names = X_train_tree.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importance (Closer to Root = Higher Score)")
plt.bar(range(X_train_tree.shape[1]), importances[indices], align="center", color='steelblue')
plt.xticks(range(X_train_tree.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.ylabel('Importance Score (Gini Impurity Decrease)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], 
          feature_names = feature_names,
          class_names = [str(c) for c in LE.classes_],
          filled = True, 
          rounded = True, 
          max_depth = 3,
          fontsize = 10)
plt.title("Top 3 Levels of the First Decision Tree (Root Distribution)")
plt.show()

print(f"> Most important feature (Root Candidate): {feature_names[indices[0]]}")

print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")



# SVM training
sampleIndex = np.random.choice(X_train_lin.index, size=int(len(X_train_lin) * 0.1), replace = False)
Xsplit = X_train_lin.loc[sampleIndex]
ysplit = y_train.loc[sampleIndex]
print(f"\n> Tuning SVM on subset of {len(Xsplit)} rows...")
 
Cs = [0.001, 0.01, 0.1, 1, 10, 100]
kernels = ['rbf', 'linear', 'sigmoid']
kernelResults = {}
bestScore = -1
bestParams = {}
print(f"\n\n> SVM training: evaluate SVM classifiers with C ranging inside {Cs}")
for kernel in kernels:   
    scores = []
    print(f"\n> Evaluating: {kernel.upper()}")
    for C in Cs:
        SVCPipeline = make_pipeline(StandardScaler(), SVC(C = C, kernel = kernel, random_state = randomState, class_weight = 'balanced'))
        cv_scores = cross_val_score(SVCPipeline, Xsplit, ysplit, cv = 5, scoring = 'f1_macro')
        mean_score = cv_scores.mean()
        scores.append(mean_score)
        print(f"  > C = {C:<5} | F1-score = {mean_score:.4f}")
        if mean_score > bestScore:
            bestScore = mean_score
            bestParams = {'C': C, 'kernel': kernel}
    kernelResults[kernel] = scores

print(f"\n> Generating Kernel Comparison Graph")
plotKernelPerformanceComparison(kernelResults, Cs)

bestC = bestParams['C']
bestKernel = bestParams['kernel']
print(f"\n> Best SVM Params: Kernel='{bestKernel}', C={bestC}. Retraining on FULL linear data...")

bestSVCPipeline = make_pipeline(StandardScaler(), SVC(C = bestC, kernel = bestKernel, random_state = randomState, class_weight = 'balanced'))
bestSVCPipeline.fit(X_train_lin, y_train)
predVal = bestSVCPipeline.predict(X_test_lin)

displayConfusionMatrix(y_test, predVal, f"Best SVM (HP = C={bestC}, kernel={bestKernel})")

print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")


# LR training with PCA
LR_Cs = [0.001, 0.01, 0.1, 1, 10, 100]
LRcrossValidationScores = []

print(f"\n\n> Logistic Regression (with PCA) training: evaluate LRs with C ranging inside {LR_Cs}")
for C in LR_Cs:
    LRPipeline = make_pipeline(
        StandardScaler(), 
        PCA(n_components=0.95, random_state=randomState), 
        LogisticRegression(C = C, random_state = randomState, max_iter = 1000, class_weight = 'balanced')
    )
    scores = cross_val_score(LRPipeline, X_train_lin, y_train, cv = 5, scoring = 'f1_macro')
    LRcrossValidationScores.append(scores.mean())
    print(f"> C = {C}, F1-score (mean of batch) = {scores.mean():.4f}")

bestLR_C = LR_Cs[np.argmax(LRcrossValidationScores)]
print(f"\n> Best Logistic Regression C: {bestLR_C}. Retraining...")

bestLRPipeline = make_pipeline(
    StandardScaler(), 
    PCA(n_components=0.95, random_state=randomState),
    LogisticRegression(C = bestLR_C, random_state = randomState, max_iter = 1000, class_weight = 'balanced')
)
bestLRPipeline.fit(X_train_lin, y_train)
predVal = bestLRPipeline.predict(X_test_lin)

plotLogQualityCheckGraph(LRcrossValidationScores, LR_Cs, bestLR_C, max(LRcrossValidationScores), "Logistic Regression (with PCA)")
displayConfusionMatrix(y_test, predVal, f"Best Logistic Regression (C = {bestLR_C})")

print(f"Accuracy: {accuracy_score(y_test, predVal):.4f}")
print(f"Precision: {precision_score(y_test, predVal, average = 'macro'):.4f}")
print(f"Recall: {recall_score(y_test, predVal, average = 'macro'):.4f}")



## Clustering
print(f"\n\n> Clustering models to be implemented:\n1. KMeans\n2. GMM\nBoth models will be trained on \"raw\" data and on PCA-optimized data to compare results")

# For clustering, we use X_linear (Colors) as it represents the spectral type better.
X_clustering = X_linear
print(f"> Using Linear (Color) features for clustering: {X_clustering.columns.to_list()}")

# Traing new scaler for clustering
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_clustering), columns = X_clustering.columns)
PCA_clust = PCA(n_components = 2, random_state = randomState)
X_PCA = PCA_clust.fit_transform(X_scaled)


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
axes[0, 0].set_title('KMeans (Trained on Standardized Data)')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
sns.scatterplot(x = X_PCA[:, 0], y = X_PCA[:, 1], hue = PCA_Kmeans, palette = 'viridis', alpha = 0.6, ax = axes[0, 1])
axes[0, 1].set_title('KMeans (Trained on PCA Data)')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')

# GMM Plots
sns.scatterplot(x = X_PCA[:, 0], y = X_PCA[:, 1], hue = raw_GMM, palette = 'magma', alpha = 0.6, ax = axes[1, 0])
axes[1, 0].set_title('GMM (Trained on Standardized Data)')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
sns.scatterplot(x = X_PCA[:, 0], y = X_PCA[:, 1], hue = PCA_GMM, palette = 'magma', alpha = 0.6, ax = axes[1, 1])
axes[1, 1].set_title('GMM (Trained on PCA Data)')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')

plt.tight_layout()
plt.show()

print("\n\n> Clustering Quantitative Evaluation")
printClusterMetrics("KMeans (Std Data)", y, raw_Kmeans, X_scaled)
printClusterMetrics("KMeans (PCA Data)", y, PCA_Kmeans, X_PCA)
printClusterMetrics("GMM (Std Data)", y, raw_GMM, X_scaled)
printClusterMetrics("GMM (PCA Data)", y, PCA_GMM, X_PCA)