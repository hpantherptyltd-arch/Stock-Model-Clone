import pandas as pd
import numpy as np

def create_df(ticker):
    
    df = pd.read_csv(f"data/{ticker}_10y.csv", skiprows=3)
    
    df.columns = [
        "Date",
        "AdjClose",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)
    return df

def make_windows(data, window, horizon):
    X, Y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i:i+window])
        Y.append(data[i+window:i+window+horizon])
    return np.array(X), np.array(Y)