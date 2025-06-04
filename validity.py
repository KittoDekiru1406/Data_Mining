# Indices of Cluster Validity
import numpy as np
import math
# from ds.clustering.utility import norm_distances

def norm_distances(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist
    return cdist(XA, XB)
    # return np.sqrt(((XA[:, np.newaxis, :] - XB) ** 2).sum(axis=2))

# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

# DI fast
def dunn_fast(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics.pairwise import euclidean_distances

    def __delta_fast(ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]
        return np.min(values)

    def __big_delta_fast(ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        # values = values [np.nonzero(values)]
        return np.max(values)
    # -----------------------------------
    distances = euclidean_distances(data)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = __delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        big_deltas[k] = __big_delta_fast((labels == ks[k]), distances)
    return np.min(deltas)/np.max(big_deltas)


# DI
def dunn(data: np.ndarray, labels: np.ndarray) -> float:
    C = len(np.unique(labels))
    cluster_points = [data[labels == i] for i in range(C)]
    cluster_centers = np.array([np.mean(points, axis=0) for points in cluster_points])
    # Tính khoảng cách nhỏ nhất giữa các tâm cụm
    min_cluster_distance = np.inf
    from itertools import combinations
    for i, j in combinations(range(C), 2):
        dist = norm_distances(cluster_centers[i], cluster_centers[j])
        min_cluster_distance = min(min_cluster_distance, dist)
    # Tính đường kính lớn nhất của các cụm
    max_cluster_diameter = 0
    for points in cluster_points:
        if len(points) > 1:  # Cụm phải có ít nhất 2 điểm để tính đường kính
            distances = norm_distances(points[:, np.newaxis], points, axis=2)
            cluster_diameter = np.max(distances)
            max_cluster_diameter = max(max_cluster_diameter, cluster_diameter)
    if max_cluster_diameter == 0:
        return np.inf
    return min_cluster_distance / max_cluster_diameter


# DB
def davies_bouldin(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(data, labels)


# PC fuzzy
def partition_coefficient(membership: np.ndarray) -> float:
    N = membership.shape[0]
    return np.sum(np.square(membership)) / N


# CE fuzzy
def classification_entropy(membership: np.ndarray, a: float = np.e) -> float:
    N = membership.shape[0]

    epsilon = np.finfo(float).eps
    membership = np.clip(membership, epsilon, 1)

    log_u = np.log(membership) / np.log(a)  # Chuyển đổi cơ số logarit
    return -np.sum(membership * log_u) / N


# PE fuzzy
def partition_entropy(membership: np.ndarray) -> float:
    return classification_entropy(membership=membership, a=np.e)


def purity_score(membership: np.ndarray) -> float:
    return np.mean([np.max(membership[i]) for i in range(len(membership))])


# S fuzzy
def separation(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    _N, C = membership.shape
    _ut = membership.T
    numerator = 0
    for i in range(C):
        diff = data - centroids[i]
        squared_diff = np.sum(diff**2, axis=1)
        numerator += np.sum((_ut[i] ** m) * squared_diff)
    center_dists = np.sum((centroids[:, np.newaxis] - centroids) ** 2, axis=2)
    np.fill_diagonal(center_dists, np.inf)
    min_center_dist = np.min(center_dists)
    return numerator / min_center_dist


# CH
def calinski_harabasz(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(data, labels)


# SI
def silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    return silhouette_score(data, labels)


# AC
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


# F1
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average)


# FHV fuzzy
def hypervolume(membership: np.ndarray, m: float = 2) -> float:
    C = membership.shape[1]
    result = 0
    for i in range(C):
        cluster_u = membership[:, i]
        n_i = np.sum(cluster_u > 0)
        if n_i > 0:
            result += np.sum(cluster_u ** m) / n_i
    return result


# CS fuzzy
def cs(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    N, C = membership.shape
    numerator = 0
    for i in range(C):
        numerator += np.sum((membership[:, i]**m)[:, np.newaxis] *
                            np.sum((data - centroids[i])**2, axis=1)[:, np.newaxis])
    min_center_dist = np.min([np.sum((centroids[i] - centroids[j])**2)
                              for i in range(C)
                              for j in range(i+1, C)])
    return numerator / (N * min_center_dist)


# XB fuzzy
def Xie_Benie(data: np.ndarray, centroids: np.ndarray, membership: np.ndarray) -> float:
    _N, C = membership.shape
    labels = np.argmax(membership, axis=1)
    clusters = [data[labels == i] for i in range(C)]

    from sklearn.metrics import pairwise_distances
    S_iq = np.asanyarray([np.mean([np.linalg.norm(point - centroids[i]) for point in cluster]) for i, cluster in enumerate(clusters)])
    tu = np.sum(np.square(membership) * np.square(S_iq))
    distance = pairwise_distances(centroids)
    distance[distance == 0] = math.inf
    mau = len(data) * np.min(np.square(distance))
    return tu / mau
