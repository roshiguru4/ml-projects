# project descriptions

1. misc projects: nothing ml related, just decided to put in here for convenience.
2. nba longevity logistic regression model
  - scraped basketball-references.com for career data using beautiful soup and requests
  - cleaned data using pandas framework
  - used scikit-learn package to model 12 features (pts, ast, reb, stl, 3p%...) w/ 80/20 train/test split
  - 77.2% accuracy to determine 'long' nba career (>5 years)
3. obesity random forest classification
  -  donwloaded .csv file from UCI ml repository
  -  used scikit-learn encoder module to change categorical data to numerical format
  -  modeled data w/ RFC (100 estimators & 17+ features) w/ 80/20 train/test split
  -  achieved 95.5% accuracy on determining obesity type and level
4. s&p 500 risk analysis
  - scraped tickers from wikipedia and used yfinance python library for collecting relevant risk indicators
  - determined avg. risk score using percentile and fixed thresholds (buy, hold, sell) -> (0, 1, 2)
  - used RFC model (100 estimators & 9 features)
  - achieved 87.7% accuracy on determining buy, hold, or sell on s&p 500 stocks


