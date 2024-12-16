import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

def load_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

nvda_data = load_stock_data('NVDA', '2020-01-01', '2024-12-01')
nvdq_data = load_stock_data('NVDQ', '2020-01-01', '2024-12-01')

def add_features(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
    df['Bollinger_Upper'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['Bollinger_Lower'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Change'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

def prepare_data(df, lookback=5):
    df = add_features(df)
    for i in range(1, lookback + 1):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

nvda_prepared = prepare_data(nvda_data)
nvdq_prepared = prepare_data(nvdq_data)

combined_data = pd.concat([nvda_prepared, nvdq_prepared], axis=1, keys=['NVDA', 'NVDQ'])

train_size = int(0.8 * len(combined_data))
train_data = combined_data.iloc[:train_size]
test_data = combined_data.iloc[train_size:]

numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()

imputer = SimpleImputer(strategy='mean')
train_data[numeric_columns] = imputer.fit_transform(train_data[numeric_columns])
test_data[numeric_columns] = imputer.transform(test_data[numeric_columns])

train_scaled = scaler.fit_transform(train_data[numeric_columns])
test_scaled = scaler.transform(test_data[numeric_columns])

lookback = 5
X_train = []
y_train = []
X_test = []
y_test = []

for i in range(lookback, len(train_scaled)):
    X_train.append(train_scaled[i - lookback:i, :])
    y_train.append(train_scaled[i, 0])  

for i in range(lookback, len(test_scaled)):
    X_test.append(test_scaled[i - lookback:i, :])
    y_test.append(test_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, test_scaled[lookback:, 1:]), axis=1))[:, 0]
actuals = scaler.inverse_transform(test_scaled)[:, 0][lookback:]

def prepare_ensemble_features(predictions, test_data, lookback):
    predictions_feature = predictions.reshape(-1, 1)
    additional_features = test_data[lookback:, :]
    ensemble_features = np.concatenate((predictions_feature, additional_features), axis=1)
    return ensemble_features

ensemble_features = prepare_ensemble_features(predictions, test_scaled, lookback)

rf_scaler = StandardScaler()
X_ensemble = rf_scaler.fit_transform(ensemble_features)
y_ensemble = actuals

train_size = int(0.8 * len(X_ensemble))
X_rf_train, X_rf_test = X_ensemble[:train_size], X_ensemble[train_size:]
y_rf_train, y_rf_test = y_ensemble[:train_size], y_ensemble[train_size:]

rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf_model.fit(X_rf_train, y_rf_train)

gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=7, random_state=42)
gb_model.fit(X_rf_train, y_rf_train)

hgb_model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=7, random_state=42)
hgb_model.fit(X_rf_train, y_rf_train)

rf_predictions = rf_model.predict(X_rf_test)
gb_predictions = gb_model.predict(X_rf_test)
hgb_predictions = hgb_model.predict(X_rf_test)

final_predictions = (0.4 * rf_predictions + 0.4 * gb_predictions + 0.2 * hgb_predictions)

mae_lstm = mean_absolute_error(actuals, predictions)
mse_lstm = mean_squared_error(actuals, predictions)
r2_lstm = r2_score(actuals, predictions)

mae_ensemble = mean_absolute_error(y_rf_test, final_predictions)
mse_ensemble = mean_squared_error(y_rf_test, final_predictions)
r2_ensemble = r2_score(y_rf_test, final_predictions)

plt.figure(figsize=(10, 6))
plt.plot(y_rf_test, label='Actual Prices', alpha=0.7)
plt.plot(final_predictions, label='Final Predictions (Ensemble)', alpha=0.7)
plt.title('NVDA Stock Price Prediction - Improved Ensemble Model')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

print("LSTM Model:")
print(f"Mean Absolute Error: {mae_lstm}")
print(f"Mean Squared Error: {mse_lstm}")
print(f"R² Score: {r2_lstm}")

print("Improved Ensemble Model:")
print(f"Mean Absolute Error: {mae_ensemble}")
print(f"Mean Squared Error: {mse_ensemble}")
print(f"R² Score: {r2_ensemble}")