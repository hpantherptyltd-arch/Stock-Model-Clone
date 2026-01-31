import yfinance as yf

# Choose the tickers
tickers = ["RKLB","ASTS","AMZN","NBIS","GOOGL","RDDT","MU","SOFI","POET","AMD","IREN","HOOD","RIVM","NVDA","ONDS","LUNR","APLD","TSLA","PLTR","META","NVO","AVGO","PATH","PL","NFLX","OPEN","ANIC","TMC","FNMA","UBER"]

for t in tickers:
    # Download historical data
    df = yf.download(
        t,
        start="2015-01-01",
        end="2025-01-01",
        auto_adjust=False
    )

    # Save to CSV
    df.to_csv(f"stock/data/{t}_10y.csv")

print("CSV files saved successfully.")