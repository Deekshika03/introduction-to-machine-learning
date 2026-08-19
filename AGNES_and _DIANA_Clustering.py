# Experiment: AGNES and DIANA Clustering

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage


# Step 2: Load Iris Dataset
iris = load_iris()

X = iris.data

print("Dataset Shape:", X.shape)
print("Features:", iris.feature_names)


# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================================
# AGNES - Agglomerative Nesting
# ==========================================================

# Step 4: Create dendrogram
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))

dendrogram(
    linked,
    truncate_mode='level',
    p=5
)

plt.title("AGNES Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


# Step 5: Apply AGNES
agnes = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

agnes_labels = agnes.fit_predict(X_scaled)


# Step 6: Calculate AGNES Silhouette Score
agnes_score = silhouette_score(
    X_scaled,
    agnes_labels
)

print("\nAGNES Results")
print("Number of Clusters:", 3)
print("Silhouette Score:", round(agnes_score, 4))


# Step 7: Visualize AGNES clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=agnes_labels
)

plt.title("AGNES Clusters")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# ==========================================================
# DIANA - Divisive Analysis
# ==========================================================

# Step 8: Define DIANA function
def diana(X, n_clusters=3):

    # Initially all points belong to one cluster
    clusters = [list(range(len(X)))]

    while len(clusters) < n_clusters:

        # Select the largest cluster
        cluster = max(clusters, key=len)
        clusters.remove(cluster)

        # Cannot split a single-point cluster
        if len(cluster) <= 1:
            clusters.append(cluster)
            break

        # Calculate centroid
        centroid = np.mean(X[cluster], axis=0)

        # Calculate distance of each point from centroid
        distances = [
            np.linalg.norm(X[i] - centroid)
            for i in cluster
        ]

        # Find the most dissimilar point
        split_point = cluster[np.argmax(distances)]

        # Create new cluster
        new_cluster = [split_point]

        remaining_cluster = [
            i for i in cluster
            if i != split_point
        ]

        # Move points closer to split point
        for i in remaining_cluster.copy():

            distance_old = np.linalg.norm(
                X[i] - centroid
            )

            distance_new = np.linalg.norm(
                X[i] - X[split_point]
            )

            if distance_new < distance_old:
                new_cluster.append(i)
                remaining_cluster.remove(i)

        # Store the two new clusters
        clusters.append(remaining_cluster)
        clusters.append(new_cluster)

    # Assign cluster labels
    labels = np.zeros(len(X), dtype=int)

    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id

    return labels


# Step 9: Apply DIANA
diana_labels = diana(
    X_scaled,
    n_clusters=3
)


# Step 10: Calculate DIANA Silhouette Score
diana_score = silhouette_score(
    X_scaled,
    diana_labels
)

print("\nDIANA Results")
print("Number of Clusters:", 3)
print("Silhouette Score:", round(diana_score, 4))


# Step 11: Visualize DIANA clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=diana_labels
)

plt.title("DIANA Clusters")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# ==========================================================
# COMPARISON OF AGNES AND DIANA
# ==========================================================

# Step 12: Create comparison table
results = pd.DataFrame({
    "Method": ["AGNES", "DIANA"],
    "Number of Clusters": [3, 3],
    "Silhouette Score": [
        agnes_score,
        diana_score
    ]
})

print("\nClustering Comparison:")
print(results)
