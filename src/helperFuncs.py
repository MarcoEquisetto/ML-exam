import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

def drawCorrelationMatrix(dataset):
    plt.figure(figsize=(12, 10))
    CM = dataset.corr(numeric_only=True).abs().style.background_gradient(axis=None, cmap='Reds')
    sns.heatmap(CM.data, annot=True, fmt=".2f", cmap='Reds')
    plt.title('Feature Correlation Matrix')
    plt.show()
    return CM

def plotQualityCheckGraph(scores, bestHyperparameter, bestScore, modelName):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(scores)+1), scores, marker='o', markersize=4, color='steelblue')
    plt.title(f'{modelName}: F1-Score vs. HyperParameter (Best = {bestHyperparameter})', fontsize=14)
    plt.xlabel('HyperParameter Value')
    plt.ylabel('F1-Score (macro)')
    plt.grid(True, alpha=0.3)
    plt.axvline(bestHyperparameter, color='red', linestyle='--', linewidth=2, label=f'Best HP = {bestHyperparameter}')
    plt.scatter(bestHyperparameter, bestScore, color='red', s=100, zorder=5)
    plt.legend()
    plt.show()

def printClusterMetrics(modelName, trueLabels, clusterLabel, clusteringData):
    ari = adjusted_rand_score(trueLabels, clusterLabel)
    nmi = normalized_mutual_info_score(trueLabels, clusterLabel)
    sil = silhouette_score(clusteringData, clusterLabel, sample_size=5000)
    
    print(f"\n>MODEL: {modelName}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f} (1.0 is perfect)")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f} (1.0 is perfect)")
    print(f"Silhouette Score: {sil:.4f} (Higher is better)\n")


def plotLogQualityCheckGraph(scores, params, bestHyperparameter, bestScore, modelName):
    fig = plt.figure(figsize=(10, 5))
    
    plt.semilogx(params, scores, marker='o', markersize=6, color='darkorange', linewidth=2)
    
    plt.title(f'{modelName}: F1-Score vs. Regularization C (Best = {bestHyperparameter})', fontsize=14)
    plt.xlabel('Regularization Parameter C (Log Scale)')
    plt.ylabel('F1-Score (macro)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.axvline(bestHyperparameter, color='red', linestyle='--', linewidth=2, label=f'Best C = {bestHyperparameter}')
    plt.scatter(bestHyperparameter, bestScore, color='red', s=150, zorder=5, edgecolors='black')
    
    plt.legend()
    plt.show()