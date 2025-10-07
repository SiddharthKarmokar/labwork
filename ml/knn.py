import numpy as np
import pandas as pd
import sys

sys.stdin = open("1.in", "r")

n = int(input())
data = []
for _ in range(n):
    a, b, c, d, e = input().split()
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    data.append((a, b, c, d, e))

df = pd.DataFrame(data, columns=['sepallen', 'sepalwidth', 'petallen', 'petalwidth', 'class'])
X = df.drop(columns=['class'])
Y = df['class']


X = (X - X.min())/(X.max() - X.min())
Y_label, Y_classes = pd.factorize(Y)


def manhatten_dist(a1, b1, c1, d1, a2, b2, c2, d2):
    a = abs(a1-a2)
    b = abs(b1-b2)
    c = abs(c1-c2)
    d = abs(d1-d2)
    return a+b+c+d

def knn(X_train, Y_train, X_test, Y_test, k):
    acc = 0
    for index, row in enumerate(X_test.itertuples(index=False)):
        tp = []
        for idx, row1 in enumerate(X_train.itertuples(index=False)):
            dist = float(manhatten_dist(row[0], row[1], row[2], row[3], row1[0], row1[1], row1[2], row1[3]))
            tp.append((dist, idx))
        tp.sort()
        odds = {int(c):0 for c in set(Y_train)}
        for i in range(k):
            odds[int(Y_train[tp[i][1]])] += 1
        max_odd = max(odds.items(), key=lambda x: x[1])
        if max_odd[0] == Y_test[index]:
            acc += 1
    return acc / len(X_test)

split = int(0.8*n)
acc = 0
k = 1

X_test = X[split:]
X_test = X_test.reset_index(drop=True)
Y_test = Y_label[split:]

X_train = X[:split]
X_train = X_train.reset_index(drop=True)
Y_train = Y_label[:split]
acc = knn(X_train, Y_train, X_test, Y_test, k)
print("FINAL EVALUATION ON TEST SET")
print(f"Test Set accuracy with k={k}: {acc:.2f}")