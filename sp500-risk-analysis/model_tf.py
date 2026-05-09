import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

df = pd.read_csv("sp500-risk-analysis/sp500_data.csv")

THRESHOLDS = {
    'volatility':      (df['volatility'].quantile(0.33),      df['volatility'].quantile(0.66)),
    'drawdown':        (df['drawdown'].quantile(0.33),         df['drawdown'].quantile(0.66)),
    'beta':            (1.0, 1.5),
    'market_cap':      (df['market_cap'].quantile(0.33),       df['market_cap'].quantile(0.66)),
    'pe_ratio':        (df['pe_ratio'].quantile(0.33),         df['pe_ratio'].quantile(0.66)),
    'dividend_yield':  (df['dividend_yield'].quantile(0.33),   df['dividend_yield'].quantile(0.66)),
    'profit_margin':   (df['profit_margin'].quantile(0.33),    df['profit_margin'].quantile(0.66)),
    'return_on_equity':(df['return_on_equity'].quantile(0.33), df['return_on_equity'].quantile(0.66)),
    'debt_to_equity':  (0.5, 1.5),
}

for column, (low, high) in THRESHOLDS.items():
    df[f'{column}_risk'] = np.select(
        [df[column] < low, df[column].between(low, high), df[column] > high],
        [0, 1, 2]
    )

risk_columns = [f'{col}_risk' for col in THRESHOLDS.keys()]
df['avg_risk'] = df[risk_columns].mean(axis=1)

def risk_label(avg):
    if avg < 0.75:   return 0
    elif avg < 1.5:  return 1
    else:            return 2

df['risk_label'] = df['avg_risk'].apply(risk_label)
df = df.drop(columns=risk_columns + ['avg_risk'])

features = ['volatility', 'drawdown', 'beta', 'market_cap', 'pe_ratio',
            'dividend_yield', 'profit_margin', 'return_on_equity', 'debt_to_equity']

X = df[features].values
y = df['risk_label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(9,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}  |  Test loss: {test_loss:.3f}")

y_proba = model.predict(X_test, verbose=0)
y_pred  = np.argmax(y_proba, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Sell']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("sp500-risk-analysis/risk_model.keras")
