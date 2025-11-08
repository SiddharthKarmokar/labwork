from sklearn.datasets import make_blobs
from kmeanstp import elbow_method
import numpy as np
import matplotlib.pyplot as plt
X, _ = make_blobs(n_samples=200, centers=4)

wcss = elbow_method(X, max_k=8)

diffs = np.diff(wcss)
best_k = np.argmax(diffs) + 1
print(f"Best k (elbow point): {best_k}")
plt.plot(range(1, 9), wcss, 'bo-')
plt.show()
