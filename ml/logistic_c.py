import math
import pandas as pd
import numpy as np
import sys


sys.stdin = open("1.in", 'r')

data = []
n = int(input())
for _ in range(n):
    score, admit = map(int, input().split(","))
    data.append((score, admit))

df = pd.DataFrame(data, columns=['exam_score', 'admitted'])
print("First 5 rows:")
print(df.head())
print()
print("Shape (N, d):", df.shape)
print()
min = df['exam_score'].min()
max = df['exam_score'].max()
mean = df['exam_score'].mean()
std = df['exam_score'].std(ddof=0)
print('Summary statistics for exam_score:')
print('Min:', min)
print('Max:', max)
print('Mean:', mean)
print(f'Std: {std:.2f}')
lr = 0.01
epochs = 1000
b1 = 0
b0 = 0
x = df['exam_score'].values
y = df['admitted'].values
N = len(y)
def sigmoid(z):
    return 1/( 1 + np.exp(-z))

def clip(y_p):
    eps = 1e-5
    y_p = np.clip(y_p, eps, 1 - eps)
    return y_p

def compute_loss(y, y_p):
    y_p = clip(y_p)
    return -np.mean(y*np.log(y_p) + (1 - y)*np.log(1 - y_p))

for epoch in range(epochs):
    z = b0 + b1*x
    y_pred = sigmoid(z)
    error = y_pred - y
    db1 = np.mean(error * x)
    db0 = np.mean(error)
    b1 -= lr*db1
    b0 -= lr*db0
print()
print(f"Final theta0: {b0:.2f}")
print(f"Final theta1: {b1:.2f}")
z = b0 + b1*x
y_p = sigmoid(z)
loss = compute_loss(y, y_p)
print(f"Final loss: {loss:.2f}")
print()
x_test = [65, 155]
for xt in x_test:
    z = b0 + b1*xt
    y_p = sigmoid(z)
    print(f"Prediction for exam_score={xt}: {y_p:.2f}")