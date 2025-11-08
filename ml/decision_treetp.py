import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_sample_split=2, gain="gini"):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.gain = gain
    
    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def predict(self, X):
        pass

    def _build(self, X, y, depth):
        if depth > self.max_depth or (len(np.unique(y)) == 1) or (len(y) < self.min_sample_split):
            return np.bincount(y).argmax()
        split = self._best_split(X, y)
        if split is None:
            return np.bincount(y).argmax()
        j, t = split
        left_mask = X[:, j] <= t
        return {
            "feature": j,
            "threshold": t,
            "left" : self._build(X[left_mask], y[left_mask], depth + 1),
            "right" : self._build(X[~left_mask], y[~left_mask], depth + 1)
        }
            
        
    def _best_split(self, X, y):
        min_g, best_split = 1, None
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                left, right = y[X[:, j] <= t], y[X[:, j] > t] 
                if self.gain == "gini":
                    g = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                else:
                    g = (len(left) * self._entropy(left) + len(right) * self._entropy(right)) / len(y)
                if g < min_g:
                    min_g = g
                    best_split = (j, t)
        return best_split

    def _traverse(self):
        pass


    def _gini(self, x):
        p = np.bincount(x) / len(x)
        return 1 - np.sum(p ** 2)

    def _entropy(self, x):
        p = np.bincount(x) / len(x)
        p = p[p>0]
        return -np.sum(p * np.log2(p))
