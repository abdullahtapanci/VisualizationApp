"""HDBSCAN guest segmentation.

Density-based clustering — unlike k-means it picks its own number of
clusters and labels low-density points as noise (-1). Tune
`min_cluster_size` to the smallest persona you'd accept.
"""
# macOS filesystem is case-insensitive, so a script literally named
# HDBSCAN.py can shadow `import hdbscan`. Drop our own dir from
# sys.path before importing the package, then put it back so the
# sibling features.py still resolves.
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from features import build_stay_features, numeric_matrix  # noqa: E402

OUT = Path(_HERE)

feats = build_stay_features()
X, names = numeric_matrix(feats)
Xs = StandardScaler().fit_transform(X)
print(f"stays: {len(feats)}, features: {len(names)}")

# min_cluster_size = smallest persona you'd accept; min_samples controls
# how conservative the noise label is (higher => more noise).
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(Xs)

unique = sorted(set(labels))
n_clusters = len([c for c in unique if c != -1])
n_noise = int((labels == -1).sum())
print(f"clusters: {n_clusters}, noise: {n_noise}/{len(labels)}")

ct = pd.crosstab(feats["true_persona"], pd.Series(labels, index=feats.index, name="cluster"))
print("\npersona x cluster (-1 = noise):")
print(ct)

# per-cluster mean of original features (interpretable)
profile = (
    pd.DataFrame(X, index=feats.index, columns=names)
    .assign(cluster=labels)
    .groupby("cluster")
    .mean()
)
profile.to_csv(OUT / "hdbscan_profiles.csv")

pca = PCA(n_components=2).fit_transform(Xs)
fig, ax = plt.subplots(figsize=(8, 6))
mask = labels == -1
ax.scatter(pca[mask, 0], pca[mask, 1], c="lightgrey", s=10, label="noise")
sc = ax.scatter(pca[~mask, 0], pca[~mask, 1], c=labels[~mask], cmap="tab10", s=14)
ax.set(title=f"HDBSCAN: {n_clusters} clusters, {n_noise} noise", xlabel="PC1", ylabel="PC2")
ax.legend(loc="best")
plt.tight_layout()
fig.savefig(OUT / "hdbscan_clusters.png", dpi=120)
print(f"\nsaved {OUT / 'hdbscan_clusters.png'}")
print(f"saved {OUT / 'hdbscan_profiles.csv'}")
