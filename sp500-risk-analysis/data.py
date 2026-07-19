import io

import numpy as np
import pandas as pd
import requests
import yfinance
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CSV_PATH = 'sp500_data.csv'

NUMERIC_COLUMNS = [
    'volatility', 'drawdown', 'beta', 'market_cap', 'pe_ratio',
    'dividend_yield', 'profit_margin', 'return_on_equity', 'debt_to_equity',
]


def fetch_from_yfinance():
    """Optional live ingestion: scrape the S&P 500 list and pull metrics per
    ticker from Yahoo Finance, returning a list of row dicts.

    Kept for refreshing sp500_data.csv when you want new data. The default
    pipeline in main() reads the local CSV instead, so no network is required.
    Fetching stays plain Python: it's ~500 sequential network calls, which
    Spark's distributed engine doesn't speed up.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Wikipedia rejects requests with no User-Agent (403), so fetch manually.
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    sp500_df = pd.read_html(io.StringIO(html))[0]
    tickers = [t.replace('.', '-') for t in sp500_df['Symbol'].tolist()]

    results = []
    for ticker in tickers:
        stock = yfinance.Ticker(ticker)
        hist = stock.history(period='30d')
        info = stock.info

        returns = hist['Close'].pct_change().dropna()
        volatility = float(np.std(returns))
        drawdown = float((hist['Close'].max() - hist['Close'].min()) / hist['Close'].max())

        results.append({
            'tickers': ticker,
            'volatility': volatility,
            'drawdown': drawdown,
            'beta': info.get('beta'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield'),
            'profit_margin': info.get('profitMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
        })
        print(f'{ticker} done.')
    return results


def main():
    spark = (
        SparkSession.builder.appName('sp500-ingestion')
        .master('local[*]')
        # Pin the driver to localhost; otherwise Spark can fail to bind on macOS.
        .config('spark.driver.bindAddress', '127.0.0.1')
        .config('spark.driver.host', '127.0.0.1')
        .getOrCreate()
    )

    # ---- ingest from the local CSV via PySpark ----
    df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
    print(f'Ingested {df.count()} rows from {CSV_PATH} via PySpark.')

    # Impute missing numeric values with the column mean (Spark-native).
    means = df.select(
        [F.mean(F.col(c)).alias(c) for c in NUMERIC_COLUMNS]
    ).first().asDict()
    df = df.fillna(means)

    df.show(5)

    df.toPandas().to_csv(CSV_PATH, index=False)
    spark.stop()


if __name__ == '__main__':
    main()
