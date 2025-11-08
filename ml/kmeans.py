import random
import numpy as np

#decide clusters
#select random centroids 
#assign clusters
#move centroids 
#check finish

class Kmeans:
    def __init__(self, n_clusters: int = 2, max_iters:int = 100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centriods = None
        self.wcss = 0
    
    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centriods = X[random_index]
        cluster_group = None

        for i in range(self.max_iters):
            cluster_group = self.assign_clusters(X)
            old_centriods = self.centriods
            self.centriods = self.move_centriods(X, cluster_group)
            if np.allclose(old_centriods, self.centriods):
                break
        self.wcss = self.calc_wcss(X, cluster_group)
        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        distances = []
        for row in X:
            for centroid in self.centriods:
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()
        return np.array(cluster_group)
    
    def move_centriods(self, X, cluster_group):
        cluster_type = np.unique(cluster_group)
        new_centriods = []
        for type in cluster_type:
            new_centriods.append(X[cluster_group == type].mean(axis = 0))
        return np.array(new_centriods)
    
    def calc_wcss(self, X, cluster_group):
        wcss = 0
        for k in range(self.n_clusters):
            points = X[cluster_group == k]
            centriod = self.centriods[k]
            wcss += np.sum((points - centriod) ** 2)
        return wcss


def elbow_method(X, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        km = Kmeans(n_clusters=k)
        km.fit_predict(X)
        wcss.append(km.wcss)
    return wcss