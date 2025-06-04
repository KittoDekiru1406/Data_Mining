import time
import numpy as np
from utility import euclidean_cdist, extract_labels, extract_clusters


class Dfcm():

    def __init__(self, n_clusters: int, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000):
        if m <= 1:
            raise RuntimeError('m>1')
        self._n_clusters = n_clusters
        self._m = m
        self._epsilon = epsilon
        self._max_iter = max_iter
        self.process_time = 0

        self.local_data = None
        self.membership = None
        self.centroids = None

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    # Khởi tạo ma trận thành viên
    def __init_membership(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if seed > 0:
            np.random.seed(seed=seed)
        U0 = np.random.rand(n_samples, self._n_clusters)
        return U0 / U0.sum(axis=1)[:, None]

    # Cập nhật ma trận tâm cụm
    def _update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        # Nhân ma trận X với từng độ thuộc của nó
        # Tổng kết quả cho toàn bộ các điểm của mỗi tâm cụm
        _umT = (membership ** self._m).T
        V = ((_umT[:, :, None]) * data).sum(axis=1)
        # Chia mỗi tâm cụm cho tổng giá trị độ thuộc của các điểm thuộc trọng tâm
        return V / ((_umT).sum(axis=1)[:, None])

    @staticmethod
    def calculate_membership_by_distances(distances: np.ndarray, m: float = 2) -> np.ndarray:
        power = 2 / (m - 1)
        distances[distances == 0] = np.finfo(float).eps
        U = distances[:, :, None] * ((1 / distances)[:, None, :])
        U = (U ** power).sum(axis=2)
        return 1 / U

    # Cập nhật ma trận độ thuộc
    def calculate_membership(self, distances: np.ndarray) -> np.ndarray:
        return self.calculate_membership_by_distances(distances=distances, m=self._m)

    def update_membership(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        _sdistances = euclidean_cdist(data, centroids)  # Khoảng cách Euclidean giữa data và centroids
        return self.calculate_membership(_sdistances)

    def fit(self, data: np.ndarray, seed: int = 42) -> tuple:
        _start_tm = time.time()
        self.local_data = data
        self.membership = self.__init_membership(n_samples=len(data), seed=seed)
        for step in range(self._max_iter):
            old_u = self.membership.copy()
            self.centroids = self._update_centroids(data, old_u)
            self.membership = self.update_membership(data, self.centroids)
            if (np.abs(self.membership - old_u)).max(axis=(0, 1)) < self._epsilon:
                break
        self.process_time = time.time() - _start_tm
        return self.membership, self.centroids, step + 1

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def extract_labels(self) -> np.ndarray:
        return extract_labels(membership=self.membership)

    @property
    def extract_clusters(self) -> list:
        _labels = self.extract_labels
        return extract_clusters(data=self.local_data, labels=_labels, n_clusters=self._n_clusters)
