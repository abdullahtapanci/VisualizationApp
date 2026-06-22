"""K-means guest segmentation.

Builds per-stay features, standardizes, sweeps k for the elbow and
silhouette score, fits the best k, and saves a 2D PCA scatter plus
a crosstab against the synthetic persona labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from features import build_stay_features, numeric_matrix

OUT = Path(__file__).resolve().parent

feats = build_stay_features()
X, names = numeric_matrix(feats)
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
print(f"stays: {len(feats)}, features: {len(names)}")

ks = list(range(2, 11))
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, km.labels_))
    print(f"  k={k:2d}  inertia={km.inertia_:.1f}  silhouette={sils[-1]:.3f}")

k_best = ks[int(np.argmax(sils))]
print(f"best k by silhouette: {k_best}")

km = KMeans(n_clusters=k_best, n_init=10, random_state=0).fit(Xs)
labels = km.labels_

# how clusters line up with the synthetic personas
ct = pd.crosstab(feats["true_persona"], pd.Series(labels, index=feats.index, name="cluster"))
print("\npersona x cluster:")
print(ct)

# cluster centroids in the original feature space (interpretable)
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=names)
centers.index.name = "cluster"
centers.to_csv(OUT / "kmeans_centroids.csv")

pca = PCA(n_components=2).fit_transform(Xs)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(ks, inertias, "o-")
axes[0].set(title="Elbow", xlabel="k", ylabel="inertia")
axes[1].plot(ks, sils, "o-")
axes[1].set(title="Silhouette", xlabel="k", ylabel="score")
axes[2].scatter(pca[:, 0], pca[:, 1], c=labels, cmap="tab10", s=12)
axes[2].set(title=f"K-means k={k_best} (PCA 2D)", xlabel="PC1", ylabel="PC2")
plt.tight_layout()
fig.savefig(OUT / "kmeans_clusters.png", dpi=120)
print(f"\nsaved {OUT / 'kmeans_clusters.png'}")
print(f"saved {OUT / 'kmeans_centroids.csv'}")
