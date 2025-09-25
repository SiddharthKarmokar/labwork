import pandas as pd
import sys
sys.stdin = open('1.in', 'r')
n = int(input())
data = []
for _ in range(n):
    a, b = input().split()
    data.append((a, int(b)))
df = pd.DataFrame(data, columns=['product', 'price'])
df.sort_values(by='price')
print(df)