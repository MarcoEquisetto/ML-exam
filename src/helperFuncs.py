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

def print_cluster_metrics(model_name, true_labels, cluster_labels, data_used_for_clustering):
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    sil = silhouette_score(data_used_for_clustering, cluster_labels, sample_size=5000)
    
    print(f"\n>MODEL: {model_name}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f} (1.0 is perfect)")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f} (1.0 is perfect)")
    print(f"Silhouette Score: {sil:.4f} (Higher is better)\n")
