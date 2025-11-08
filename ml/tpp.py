import numpy as np
from decision_tree import DecisionTree

# Feature columns: [height (cm), weight (kg)]
X = np.array([
    [150, 50],
    [160, 55],
    [170, 65],
    [180, 80],
    [190, 90],
    [155, 52],
    [165, 60],
    [175, 75],
    [185, 85],
    [195, 95]
])

# Labels: 0 = Normal, 1 = Overweight
y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])

tree = DecisionTree(max_depth=3, gain="entropy")
tree.fit(X, y)

X_new = np.array([[172, 68], [158, 53], [188, 82]])
print(tree.predict(X_new))
