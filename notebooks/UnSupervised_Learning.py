import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, adjusted_rand_score
import Feature_Selection as fs 

x = fs.x_final
y = fs.y

# 1 - Apply K-Means Clustering (elbow method to determine K).
inertia_values = []
k_range = range(1, 11) 

for k in k_range:
    kmeans = KMeans(n_clusters = k,
                    random_state = 42, 
                    n_init = 10)
    kmeans.fit(x)
    inertia_values.append(kmeans.inertia_)

optimal_k = 2 
final_kmeans = KMeans(n_clusters=optimal_k, 
                      random_state=42, 
                      n_init=10)
final_kmeans.fit(x)
kmeans_labels = final_kmeans.labels_

# 2 - Perform Hierarchical Clustering (Dendrogram Analysis)
x_sample = x.sample(n=50, random_state=42)
linked_data = linkage(x_sample, method='ward')

# 3 - Compare Clusters with Actual Disease Labels
y_true = y.to_numpy()
cluster_comparison = pd.DataFrame({'KMeans Cluster': kmeans_labels, 'Actual Disease Label': y_true})
# Using Cross-Tabulation
crosstab = pd.crosstab(cluster_comparison['KMeans Cluster'], cluster_comparison['Actual Disease Label'])
# Using Adjusted Rand Index (ARI) for quantitative comparison
ari_score = adjusted_rand_score(y_true, kmeans_labels)
# Using Silhouette Score
silhouette_avg = silhouette_score(x, kmeans_labels)

if __name__ == '__main__':
    # printing the initials
    print(f'\nStarting Unsupervised Learning with {x.shape[0]} samples and {x.shape[1]} features.')

    # K-Means
    print(f'K-Means clustering completed with K = {optimal_k}.')

    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_values, marker='o', linestyle='--', color='blue')
    plt.title('K-Means Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    # Plot the Dendrogram
    plt.figure(figsize=(15, 7))
    dendrogram(
        linked_data,
        orientation='top',
        labels=x_sample.index.tolist(),
        distance_sort='descending',
        show_leaf_counts=False
    )
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, 50 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

    # Using Cross-Tabulation to visualize label distribution
    print(f'Cross-Tabulation of K-Means Clusters vs. Actual Labels')
    print(f'Rows are K-Means Clusters, Columns are Actual Labels (0=No Disease, 1=Disease)')
    print(crosstab)

    # Using ARI and Silhouette Score to visualize label distribution
    print(f'\nQuantitative Cluster Evaluation:')
    print(f'Adjusted Rand Index (ARI): {ari_score:.4f} (Closeness to true labels)')
    print(f'Silhouette Score: {silhouette_avg:.4f} (Internal cluster quality)')


    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=x['thalach'], 
        y=x['oldpeak'], 
        hue=kmeans_labels, 
        palette='viridis', 
        style=y_true,
        s=100
    )
    plt.title('K-Means Clusters vs. Actual Labels (thalach vs. oldpeak)')
    plt.xlabel('Thalach (Maximum Heart Rate Achieved)')
    plt.ylabel('Oldpeak (ST Depression induced by exercise relative to rest)')
    plt.legend(title='Cluster/Actual Label')
    plt.show()
