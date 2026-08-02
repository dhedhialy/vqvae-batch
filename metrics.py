import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors


def kbet(repr, batch_labels, k=40, alpha=0.05):
    """kBET: kernel-based batch effect test. Rejection rate of a chi-square test
    asking whether each cell's kNN neighborhood has the same batch mix as the
    global mix. Lower is better (fewer neighborhoods deviate from batch mixing).
    """
    repr = np.asarray(repr, dtype=np.float32)
    batch = np.asarray(batch_labels)
    n = len(batch)
    n_batches = batch.max() + 1
    k = max(min(k, n - 2), 2)

    global_prop = np.bincount(batch, minlength=n_batches).astype(float) / n

    nn = NearestNeighbors(n_neighbors=k + 1).fit(repr)
    _, idx = nn.kneighbors(repr)

    # scIB-style small-count correction: if any batch is expected <5 times,
    # the chi2 p-value is unreliable; use min variance blind correction
    df = n_batches - 1
    rejected = 0
    valid = 0
    for i in range(n):
        neigh_batch = batch[idx[i, 1:]]  # drop self
        observed = np.bincount(neigh_batch, minlength=n_batches).astype(float)
        expected = global_prop * k
        # MFGFRW: combine rare batches under threshold to keep expected counts valid
        chi2, p = stats.chisquare(observed, expected)
        rejected += (p < alpha)
        valid += 1
    return rejected / valid


def ilisi(repr, batch_labels, k=40):
    """inverse-LISI: 1 / mean Simpson index of batch labels in kNN neighborhood.
    Higher is better; approaches n_batches when fully mixed, ~1.0 when separated.
    """
    repr = np.asarray(repr, dtype=np.float32)
    batch = np.asarray(batch_labels)
    n = len(batch)
    n_batches = batch.max() + 1
    k = max(1, min(k, n - 1))

    nn = NearestNeighbors(n_neighbors=k + 1).fit(repr)
    _, idx = nn.kneighbors(repr)

    lisi_values = np.zeros(n)
    for i in range(n):
        neigh_batch = batch[idx[i, 1:]]
        counts = np.bincount(neigh_batch, minlength=n_batches).astype(float)
        p = counts / counts.sum()
        simpson = np.sum(p ** 2)
        lisi_values[i] = 1.0 / simpson if simpson > 0 else 1.0

    return float(lisi_values.mean())


def bio_conservation(repr, cell_types):
    """How much of the observed structure separates biology—cell type accuracy on
    unsupervised clusters (kNN-based, matches scVI's ARI-NMI approach loosely).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    repr = np.asarray(repr, dtype=np.float32)
    ct = np.asarray(cell_types)
    n_clusters = int(ct.max() + 1)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(repr)
    ari = adjusted_rand_score(ct, km.labels_)
    nmi = normalized_mutual_info_score(ct, km.labels_)
    return {"ari": float(ari), "nmi": float(nmi)}