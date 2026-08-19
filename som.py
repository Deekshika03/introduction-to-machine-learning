# Experiment: Self-Organizing Map (SOM)
# Clustering and Visualization of Digits Dataset

# Install MiniSom if required
# !pip install minisom

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from minisom import MiniSom


# 2. Load Digits Dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])


# 3. Normalize the Data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# 4. Initialize SOM
som = MiniSom(
    x=10,
    y=10,
    input_len=64,
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)

# Initialize weights
som.random_weights_init(X_scaled)


# 5. Train the SOM
som.train_random(
    X_scaled,
    10000
)

print("SOM Training Completed")


# 6. Find Best Matching Unit (BMU)
bmu_positions = []

for sample in X_scaled:
    bmu = som.winner(sample)
    bmu_positions.append(bmu)

bmu_positions = np.array(bmu_positions)


# 7. Assign Cluster Labels
cluster_labels = (
    bmu_positions[:, 0] * 10
    + bmu_positions[:, 1]
)


# 8. Calculate Silhouette Score
if len(np.unique(cluster_labels)) > 1:
    score = silhouette_score(
        X_scaled,
        cluster_labels
    )
    print("Silhouette Score:", round(score, 4))


# 9. Plot SOM Distance Map
plt.figure(figsize=(8, 8))

plt.pcolor(
    som.distance_map().T,
    cmap="bone_r"
)

plt.colorbar(label="Distance")

plt.title("SOM Distance Map")
plt.xlabel("SOM X Coordinate")
plt.ylabel("SOM Y Coordinate")

plt.show()


# 10. Visualize Digits on SOM
plt.figure(figsize=(8, 8))

plt.pcolor(
    som.distance_map().T,
    cmap="bone_r"
)

plt.colorbar(label="Distance")

markers = [
    'o', 's', 'D', '^', 'v',
    '<', '>', 'p', '*', 'h'
]

for i, sample in enumerate(X_scaled):

    x_pos, y_pos = som.winner(sample)

    plt.plot(
        x_pos + 0.5,
        y_pos + 0.5,
        markers[y[i]],
        markerfacecolor='None',
        markeredgecolor='black',
        markersize=7
    )

plt.title("Digits Distribution on SOM")
plt.xlabel("SOM X Coordinate")
plt.ylabel("SOM Y Coordinate")

plt.show()


# 11. Create Result Table
result = pd.DataFrame({
    "Actual Digit": y,
    "BMU X": bmu_positions[:, 0],
    "BMU Y": bmu_positions[:, 1],
    "Cluster Label": cluster_labels
})

print("\nSample Results:")
print(result.head(10))
