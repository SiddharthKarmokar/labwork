import sys
import numpy as np

sys.stdin = open('1.in', 'r')

n = int(input())
data = []
for _ in range(n):
    a, b, c, d = map(float, input().split())
    data.append((a, b, c, d))

data = np.array(data)

for row in data[:5]:
    print(f"{row[0]:.1f} {row[1]:.1f} {row[2]:.1f} {row[3]:.1f}")

print(data.shape)
m = data.shape[1]

stats = {
    'mean': np.mean(data, axis=0),
    'std': np.std(data, axis=0),
    'min': np.min(data, axis=0),
    'max': np.max(data, axis=0),
}
for i in range(m):
    print(f"{stats['mean'][i]:.2f} {stats['std'][i]:.2f} {stats['min'][i]:.2f} {stats['max'][i]:.2f}")

X = data[:, :-1]
y = data[:, -1]

X = (X - stats['mean'][:-1])/stats['std'][:-1]
X_d = np.c_[np.ones(X.shape[0]), X]

theta = np.zeros(X_d.shape[1])
alpha = 0.01
epochs = 300
N = len(y)

# baseline_y = np.full(y.shape, fill_value=np.mean(y))
for _ in range(epochs):
    y_pred = X_d.dot(theta)
    error = y_pred - y
    gradient = (1/N)*X_d.T.dot(error)
    theta -= alpha*gradient 
    
y_pred = X_d.dot(theta)
mse = ( np.mean((y_pred - y) ** 2) )/2

theta_ne = np.linalg.inv(X_d.T @ X_d) @ X_d.T @ y
y_pred_ne = X_d @ theta_ne
mse_ne = ( np.mean( (y_pred_ne - y)**2 ) )/2

diff_mse = abs(mse - mse_ne)
np.set_printoptions(precision=3, suppress=True)
print(f"Final theta={theta}")
print(f"Final MSE={mse:.2f}")
print(f"MSE Difference={diff_mse:.5f}")

test_x = [(150, 3, 5), (200, 4, 2)]
norm_test_x = []
for a, b, c in test_x:
    a1 = (a - stats['mean'][0])/stats['std'][0]
    b1 = (b - stats['mean'][1])/stats['std'][1]
    c1 = (c - stats['mean'][2])/stats['std'][2]
    norm_test_x.append([1, a1, b1, c1])

preds = norm_test_x @ theta
for pred in preds:
    print(f"{pred:.2f}")