import numpy as np


def dbscan(X, min_pts, eps):
    n = len(X)
    labels = np.zeros(n, dtype=int)
    cluster_id = 0

    def region_query(i):
        return np.where(np.linalg.norm(X - X[i], axis=1) < eps)[0]
    
    for i in range(n):
        if labels[i] != 0:
            continue
        neighbors = region_query(i)
        if len(neighbors) < min_pts:
            labels[i] = -1
            continue
        cluster_id += 1
        labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            p = neighbors[j]
            if labels[p] == -1:
                labels[p] = cluster_id
            elif labels[p] == 0:
                labels[p] = cluster_id
                p_neighbors = region_query(p)
                if len(p_neighbors) >= min_pts:
                    neighbors = np.concatenate((neighbors, p_neighbors))
            j += 1
    return labels
