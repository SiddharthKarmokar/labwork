import pandas as pd
import sys 

def main()->None:
    n = int(input())
    data = []
    for _ in range(n):
        a, b, c = input().split()
        data.append((int(a), b, c))
    df = pd.DataFrame(data, columns=["product_id", "low_fats", "recyclable"])
    mask = ((df["low_fats"] == "Y") & (df["recyclable"] == "Y"))    
    df1 = df.loc[mask, ["product_id"]]
    print(df1.reset_index(drop=True))
    return

if __name__ == "__main__":
    try:
        sys.stdin = open("1.in")
    except:
        pass
    main()