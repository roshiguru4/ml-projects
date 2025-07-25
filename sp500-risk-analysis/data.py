import pandas as pd
import numpy as np
import yfinance

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_df = pd.read_html(url)[0]
tickers = sp500_df['Symbol'].tolist()
tickers = [ticker.replace('.', '-') for ticker in tickers]

tickers = tickers
results = []

for ticker in tickers:
  stock = yfinance.Ticker(ticker)
  hist = stock.history(period='30d')
  info = stock.info

  returns = hist['Close'].pct_change().dropna()
  volatility = np.std(returns)
  drawdown = (hist['Close'].max() - hist['Close'].min()) / hist['Close'].max()

  results.append({
      'tickers' : ticker,
      'volatility' : volatility,
      'drawdown' : drawdown,
      'beta' : info.get('beta', None),
      'market_cap' : info.get('marketCap', None),
      'pe_ratio' : info.get('trailingPE', None),
      'dividend_yield' : info.get('dividendYield', None),
      'profit_margin' : info.get('profitMargins', None),
      'return_on_equity' : info.get('returnOnEquity', None),
      'debt_to_equity' : info.get('debtToEquity', None)
  })
  print(f'{ticker} done.')

df = pd.DataFrame(results)
df = df.fillna(df.mean(numeric_only=True))
df.to_csv('sp500_data.csv', index=False)

