import sys
import pandas as pd
import numpy as np

def helper(x, mu, sigma):
    return -(x - mu)**2/(2*(sigma**2))

def gaussian_pdf(x, mu, sigma):
    expo = helper(x, mu, sigma)
    const = 1/(sigma * np.sqrt(2*np.pi))
    return const * np.exp(expo)

def main():
    N = int(input())
    data = []
    for _ in range(N):
        data.append(tuple(map(float, input().split())))
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    train_size = int(N*0.7)
    X_train = X[:train_size]
    X_train_0 = X_train[X_train == 0]
    X_train_1 = X_train[X_train == 1]

    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    prior_1 = y_train.mean()
    mu1, mu0 = X_train_1.mean(), X_train_0.mean()
    sigma1, sigma0 = X_train_1.std(), X_train_0.std()
    
    

    


if __name__ == "__main__":
    main()