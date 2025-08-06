import pandas as pd
import sys 

def main()->None:
    n = int(input())
    data = []
    for _ in range(n):
        oid, cid, amt = map(int, input().split())
        data.append((oid, cid, amt))
    df = pd.DataFrame(data, columns=['order_id', 'customer_id', 'amount'])
    df1 = df.groupby(["customer_id"], as_index=False)["amount"].sum()
    df1.rename(columns={"amount": "total_amount"}, inplace=True)
    m = int(input())
    data = []
    for _ in range(m):
        cid, name, city = input().split()
        cid = int(cid)
        data.append((cid, name, city))
    df2 = pd.DataFrame(data, columns=['customer_id', 'name', 'city'])

    merged = pd.merge(df1, df2, on="customer_id")
    res = merged[merged["total_amount"] > 500]
    neworder = ["customer_id", "name", "city", "total_amount"]
    res = res[neworder]
    res.reset_index(drop=True, inplace=True)
    res.sort_values(by='total_amount', ascending=False)
    print(res)
    return

if __name__ == "__main__":
    try:
        sys.stdin = open("input.txt")
    except:
        pass
    main()