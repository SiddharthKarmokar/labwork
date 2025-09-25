import pandas as pd
import numpy as np
import sys

def l2_norm(y, y_pred):
    norm = 0
    for a, b in zip(y, y_pred):
        norm += (a - b)**2
    return norm

def main():
    n = int(input())
    data = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        data.append((a, b, c, d))
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values.reshape(-1, 1)
    X = (X - X.mean(axis=0))/X.std(ddof=0, axis=0)
    X = X.values
    bias = np.ones((n, 1))
    X_ = np.concatenate([X, bias], axis=1)
    d = X_.shape[1]
    W = np.zeros((d, 1))
    print(X_.shape)
    epochs = 300
    lr = 0.01
    for epoch in range(epochs):
        y_pred = X_ @ W
        grad =  ( X_.T @ (y_pred - y) )*(1/n)
        W = W - lr * grad
    W_ne = np.linalg.inv(X_.T @ X_)@(X_.T @ y) 
    y_pred = X_ @ W
    mse = (np.mean((y_pred - y)**2)) / 2
    y_pred_ne = X_ @ W_ne
    mse_ne = np.mean( (y_pred_ne - y)**2 ) / 2
    print("weight", W)
    print("mse", mse)
    print("mse ne", mse_ne)
    pass    


if __name__ == "__main__":
    try:
        sys.stdin = open('1.in', 'r')
        sys.stdout = open('1.out', 'w')
    except:
        pass
    main()