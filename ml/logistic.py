import sys
import pandas as pd
import numpy as np


epochs = 1000
lr = 0.01
threshold = 0.5
eps = 1e-5

def clip(x):
    return np.clip(x, eps, 1-eps)

def sigmoid(z):
    return np.where(z>=0, 
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def bce(y_pred, y):
    y_pred = clip(y_pred)
    return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def main():
    n = int(input())
    data = []
    for _ in range(n):
        a, b = map(int, input().split(','))
        data.append((a, b))
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(df.head())
    print(df.shape)
    print(*X.min().to_list())
    print(*X.max().to_list())
    print(*X.mean().to_list())
    print(*X.std(ddof=0).to_list())
    X = X.to_numpy().reshape(-1, 1)
    y = y.to_numpy().reshape(-1, 1)
    X_ = np.concatenate([X, np.ones((n, 1))], axis=1)
    W = np.zeros((X_.shape[1], 1))
    acc = 0
    for _ in range(epochs):
        z = X_ @ W
        y_pred = sigmoid(z)
        diff2 = (y_pred - y)
        grad = (X_.T @ diff2) / n
        W = W - lr * (grad)
    print(W)
    y_pred = sigmoid(X_ @ W)
    print(bce(y_pred, y))
    print(W.shape)
    test = [65, 100]
    for x in test:
        x_ = np.array([x, 1]).reshape((1, 2))
        y_pred = (sigmoid(x_ @ W) > threshold).astype(int)
        print(*(y_pred.tolist()))

    return

if __name__ == "__main__":
    try:
        sys.stdin = open("1.in", 'r')
        sys.stdout = open("1.out", 'w')
    except:
        pass
    main()