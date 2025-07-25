from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np  

df = pd.read_csv("sp500-risk-analysis/sp500_data.csv")
THRESHOLDS = {
    'volatility': (df['volatility'].quantile(0.33), df['volatility'].quantile(0.66)),
    'drawdown': (df['drawdown'].quantile(0.33), df['drawdown'].quantile(0.66)),
    'beta': (1.0, 1.5),
    'market_cap': (df['market_cap'].quantile(0.33), df['market_cap'].quantile(0.66)),
    'pe_ratio': (df['pe_ratio'].quantile(0.33), df['pe_ratio'].quantile(0.66)),
    'dividend_yield': (df['dividend_yield'].quantile(0.33), df['dividend_yield'].quantile(0.66)),
    'profit_margin': (df['profit_margin'].quantile(0.33), df['profit_margin'].quantile(0.66)),
    'return_on_equity': (df['return_on_equity'].quantile(0.33), df['return_on_equity'].quantile(0.66)),
    'debt_to_equity': (0.5, 1.5),
}

for column, (low, high) in THRESHOLDS.items():
    df[f'{column}_risk'] = np.select(
        [df[column] < low, df[column].between(low, high), df[column] > high],
        [0, 1, 2]
    )

risk_columns = [f'{col}_risk' for col in THRESHOLDS.keys()]
df['avg_risk'] = df[risk_columns].mean(axis=1)
def risk_label(avg):
    if avg < 0.75:
        return 0
    elif avg < 1.5:
        return 1
    else:
        return 2
df['risk_label'] = df['avg_risk'].apply(risk_label)
df = df.drop(columns=risk_columns and ['avg_risk'])

# MODEL
features = ['volatility', 'drawdown', 'beta', 'market_cap', 'pe_ratio',
            'dividend_yield', 'profit_margin', 'return_on_equity', 'debt_to_equity']
X = df[features]
y = df['risk_label']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))