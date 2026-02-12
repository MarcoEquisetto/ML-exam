
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

randomState = 42

def outlierDetection(data, n_components = 3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Fit GMM
    GMM = GaussianMixture(n_components = n_components, random_state = randomState)
    GMM.fit(X_scaled)
    
    # Get Log-Likelihood scores (Higher = Normal, Lower = Outlier)
    scores = GMM.score_samples(X_scaled)
    
    # Apply 1.5 IQR Rule
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    
    lowerBound = Q1 - (1.5 * IQR)
    mask = scores < lowerBound
    
    return mask, scores, lowerBound


def drawCorrelationMatrix(dataset):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize = (12, 10))
    CM = dataset.corr(numeric_only = True).abs().style.background_gradient(axis = None, cmap = 'Reds')
    sns.heatmap(CM.data, annot = True, fmt = ".2f", cmap = 'Reds')
    plt.title('Feature Correlation Matrix')
    plt.show()
    return CM

def plotQualityCheckGraph(scores, paramValues, bestHyperparameter, bestScore, modelName):
    import matplotlib.pyplot as plt
    plt.figure(figsize = (10, 5))
    plt.plot(paramValues, scores, marker = 'o', markersize = 4, color = 'steelblue')
    plt.title(f'{modelName}: F1-Score vs. HyperParameter (Best = {bestHyperparameter})', fontsize = 14)
    plt.xlabel('HyperParameter Value')
    plt.ylabel('F1-Score (macro)')
    plt.grid(True, alpha = 0.3)
    plt.axvline(bestHyperparameter, color = 'red', linestyle = '--', linewidth = 2, label = f'Best HP = {bestHyperparameter}')
    plt.scatter(bestHyperparameter, bestScore, color = 'red', s = 100, zorder = 5)
    plt.legend()
    plt.show()

def printClusterMetrics(modelName, trueLabels, clusterLabel, clusteringData):
    ari = adjusted_rand_score(trueLabels, clusterLabel)
    nmi = normalized_mutual_info_score(trueLabels, clusterLabel)
    sil = silhouette_score(clusteringData, clusterLabel, sample_size = 5000)
    
    print(f"\n>MODEL: {modelName}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f} (1.0 is perfect)")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f} (1.0 is perfect)")
    print(f"Silhouette Score: {sil:.4f} (Higher is better)\n")


def plotLogQualityCheckGraph(scores, params, bestHyperparameter, bestScore, modelName):
    import matplotlib.pyplot as plt
    plt.figure(figsize = (10, 5))
    
    plt.semilogx(params, scores, marker = 'o', markersize = 6, color = 'darkorange', linewidth = 2)
    
    plt.title(f'{modelName}: F1-Score vs. Regularization C (Best = {bestHyperparameter})', fontsize = 14)
    plt.xlabel('Regularization Parameter C (Log Scale)')
    plt.ylabel('F1-Score (macro)')
    plt.grid(True, which = "both", ls = "-", alpha = 0.3)
    
    plt.axvline(bestHyperparameter, color = 'red', linestyle = '--', linewidth = 2, label = f'Best C = {bestHyperparameter}')
    plt.scatter(bestHyperparameter, bestScore, color = 'red', s = 150, zorder = 5, edgecolors = 'black')
    
    plt.legend()
    plt.show()


def plotKernelPerformanceComparison(results_dict, Cs):
    import matplotlib.pyplot as plt
    plt.figure(figsize = (12, 6))
    
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (kernel_name, scores) in enumerate(results_dict.items()):
        marker = markers[idx % len(markers)]
        plt.semilogx(Cs, scores, marker = marker, linestyle = '-', linewidth = 2, label = f'Kernel: {kernel_name}')
        
        max_score = max(scores)
        max_c = Cs[np.argmax(scores)]
        plt.scatter(max_c, max_score, s = 100, edgecolors = 'black', zorder = 10, alpha = 0.6)

    plt.title('SVM Kernel Comparison: F1-Score vs Regularization (C)', fontsize = 15)
    plt.xlabel('Regularization Parameter C (Log Scale)', fontsize = 12)
    plt.ylabel('F1-Score (Macro)', fontsize = 12)
    plt.grid(True, which = "both", ls = "-", alpha = 0.3)
    plt.legend(title = "Kernels", fontsize = 10)
    plt.tight_layout()
    plt.show()


def displayConfusionMatrix(y_true, y_pred, model_name, class_names=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Greens', cbar = False,
                xticklabels = class_names if class_names else "auto",
                yticklabels = class_names if class_names else "auto")
    
    plt.xlabel('Predicted Label', fontsize = 12)
    plt.ylabel('True Label', fontsize = 12)
    plt.title(f'Confusion Matrix: {model_name}', fontsize = 14)
    plt.show()