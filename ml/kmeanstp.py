import numpy as np
import random

class Kmeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centriods = None
        self.wcss = 0
    
    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centriods = X[random_index]
        cluster_group = None
        for _ in range(self.max_iters):
            cluster_group = self._assign_clusters(X)
            old_centriods = self.centriods
            self.centriods = self._move_centriods(X, cluster_group)
            if np.allclose(old_centriods, self.centriods):
                break
        self.wcss = self._calc_wcss(X, cluster_group)
        return np.array(cluster_group)

    def _assign_clusters(self, X):
        cluster_group = []
        distances = []
        for row in X:
            for centriod in self.centriods:
                distances.append(
                    np.sqrt(np.dot(row - centriod, row - centriod))
                )
            min_distance = min(distances)
            index = distances.index(min_distance)
            cluster_group.append(index)
            distances.clear()
        return np.array(cluster_group)

    def _move_centriods(self, X, cluster_group):
        cluster_type = np.unique(cluster_group)
        new_centriod = []
        for type in cluster_type:
            new_centriod.append(X[cluster_group == type].mean(axis=0))
        return new_centriod

    def _calc_wcss(self, X, cluster_group):
        wcss = 0
        for k in range(self.n_clusters):
            points = X[cluster_group == k]
            centriod = self.centriods[k]
            wcss += np.sum((points - centriod) ** 2)
        return wcss


def elbow_method(X, max_k=8):
    wcss = []
    for k in range(1, max_k + 1):
        km = Kmeans(n_clusters=k)
        km.fit_predict(X)
        wcss.append(km.wcss)
    return wcss
