import numpy as np
import matplotlib.pyplot as plt
from dbscantp import dbscan

# --- Generate sample dataset ---
np.random.seed(42)

# Three clusters in 2D
cluster1 = np.random.randn(50, 2) + np.array([5, 5])
cluster2 = np.random.randn(50, 2) + np.array([-5, -5])
cluster3 = np.random.randn(50, 2) + np.array([5, -5])

# Add some noise points
noise = np.random.uniform(low=-10, high=10, size=(10, 2))

# Combine everything
X = np.vstack((cluster1, cluster2, cluster3, noise))

# --- Run your DBSCAN ---
labels = dbscan(X, eps=1.5, min_pts=5)

# --- Visualize results ---
plt.figure(figsize=(7, 6))
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # noise
        col = 'k'
        label = "Noise"
    else:
        label = f"Cluster {k}"
    plt.scatter(X[labels == k, 0], X[labels == k, 1], s=50, c=[col], label=label)

plt.legend()
plt.title("DBSCAN Clustering Results")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
