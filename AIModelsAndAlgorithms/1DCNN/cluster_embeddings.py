"""Run k-means and HDBSCAN on the CNN embeddings.

Drop-in companion to cnn_autoencoder.py — reads cnn_embeddings.csv,
fits both clusterers, and prints/saves the same artifacts as the
summary-stats versions so the two pipelines are directly comparable.
"""
import os
import sys
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path = [p for p in sys.path if os.path.realpath(p or ".") != _HERE]
import hdbscan  # noqa: E402
sys.path.insert(0, _HERE)

from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(_HERE)
emb = pd.read_csv(OUT / "cnn_embeddings.csv", index_col="guest_id")
persona = emb.pop("true_persona")
Z = StandardScaler().fit_transform(emb.values)
print(f"stays: {len(emb)}, embed_dim: {emb.shape[1]}")

# k-means sweep
ks, sils = list(range(2, 11)), []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    sils.append(silhouette_score(Z, km.labels_))
k_best = ks[int(np.argmax(sils))]
km = KMeans(n_clusters=k_best, n_init=10, random_state=0).fit(Z)
print(f"k-means best k={k_best}  silhouette={max(sils):.3f}")
print(pd.crosstab(persona, pd.Series(km.labels_, index=emb.index, name="kmeans")))

# HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_epsilon=0.5).fit(Z)
n_clusters = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
n_noise = int((hdb.labels_ == -1).sum())
print(f"\nHDBSCAN: {n_clusters} clusters, {n_noise} noise")
print(pd.crosstab(persona, pd.Series(hdb.labels_, index=emb.index, name="hdbscan")))

# side-by-side PCA scatter
pca = PCA(n_components=2).fit_transform(Z)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(pca[:, 0], pca[:, 1], c=km.labels_, cmap="tab10", s=12)
axes[0].set(title=f"K-means on CNN embedding (k={k_best})")
mask = hdb.labels_ == -1
axes[1].scatter(pca[mask, 0], pca[mask, 1], c="lightgrey", s=10, label="noise")
axes[1].scatter(pca[~mask, 0], pca[~mask, 1], c=hdb.labels_[~mask], cmap="tab10", s=14)
axes[1].set(title=f"HDBSCAN on CNN embedding ({n_clusters} clusters)")
axes[1].legend()
plt.tight_layout()
fig.savefig(OUT / "cnn_clusters.png", dpi=120)
print(f"\nsaved {OUT / 'cnn_clusters.png'}")
