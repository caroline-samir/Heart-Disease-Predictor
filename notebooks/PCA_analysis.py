import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import Data_PreProcessing as data_prep 

x_scaled = data_prep.df_scaled.drop('target' , axis = 1)
y = data_prep.df_scaled['target']

# 1 - Apply PCA to reduce feature dimensionality while maintaining variance.
pca = PCA()
pca.fit(x_scaled)

# 2 - Determine the optimal number of principal components using the explained variance ratio.
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

optimal_components_90 = np.where(cumulative_variance >= 0.90)[0][0] + 1 

if __name__ == '__main__':
    print(f'Total features (before PCA): {x_scaled.shape[1]}')
    print(f"\nOptimal components for 90% variance: {optimal_components_90}")
    print(f"Variance retained: {cumulative_variance[optimal_components_90 - 1]:.4f}")

    # 3 - Visualize PCA results using a scatter plot and cumulative variance plot.
    
    # 1 - cumulative variance plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle='-', label='90% Variance Threshold')
    plt.axvline(x=optimal_components_90, color='g', linestyle='--', label=f'{optimal_components_90} Components')
    plt.title('Cumulative Explained Variance Ratio (Scree Plot)')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2 - scatter plot
    pca_2d = PCA(n_components=2)
    principal_components = pca_2d.fit_transform(x_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['target'] = y.values
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df, 
                    palette='viridis', style='target', s=100)
    plt.title('2D PCA Visualization of Heart Disease Data')
    plt.xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
    plt.grid()
    plt.show()