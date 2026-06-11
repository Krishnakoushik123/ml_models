import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# Generate synthetic non-linear data
X, y = make_moons(
    n_samples=300,
    noise=0.05,
    random_state=42
)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN Model
dbscan = DBSCAN(
    eps=0.3,
    min_samples=5
)

clusters_dbscan = dbscan.fit_predict(X_scaled)

# K-Means Model
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10      # More compatible than 'auto'
)

clusters_kmeans = kmeans.fit_predict(X_scaled)

# -----------------------------
# Plotting
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -----------------------------
# K-Means Plot
# -----------------------------
scatter1 = axes[0].scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters_kmeans,
    cmap="viridis",
    marker='o',
    s=50,
    edgecolor='black'
)

axes[0].set_title("K-Means Clustering")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].grid(True)

# -----------------------------
# DBSCAN Plot
# -----------------------------

# Plot all clusters except noise
unique_labels = set(clusters_dbscan)

for label in unique_labels:

    if label == -1:
        # Noise points
        mask = clusters_dbscan == -1
        axes[1].scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            c='black',
            marker='x',
            s=70,
            label='Noise'
        )
    else:
        mask = clusters_dbscan == label
        axes[1].scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            s=50,
            label=f'Cluster {label}'
        )

axes[1].set_title("DBSCAN Clustering")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Results
# -----------------------------
n_clusters = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
n_noise = np.sum(clusters_dbscan == -1)

print("=" * 50)
print("DBSCAN Results")
print("=" * 50)
print(f"Estimated number of clusters : {n_clusters}")
print(f"Estimated number of noise points : {n_noise}")