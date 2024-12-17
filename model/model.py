from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import VotingRegressor
from tensorflow.keras.optimizers import RMSprop
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt

# Wrapper to ensure compatibility
class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period=10):
    ema = data['Close'].ewm(span=period, adjust=False).mean()
    return ema

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)

    return upper_band, rolling_mean, lower_band

def fetch_and_preprocess_data(ticker, start_date, end_date):
    extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    stock_data = yf.download(ticker, start=extended_start_date, end=end_date)

    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}. Please check the date range.")

    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['5-Day_MA'] = stock_data['Close'].rolling(5).mean()
    stock_data['10-Day_MA'] = stock_data['Close'].rolling(10).mean()
    stock_data['5-Day_Volatility'] = stock_data['Close'].rolling(5).std()

    # Lagged features
    stock_data['Lag_1'] = stock_data['Close'].shift(1)
    stock_data['Lag_2'] = stock_data['Close'].shift(2)
    stock_data['Lag_5'] = stock_data['Close'].shift(5)

    # Technical indicators
    stock_data['RSI'] = calculate_rsi(stock_data)
    stock_data['EMA_10'] = calculate_ema(stock_data)
    stock_data['Bollinger_Upper'], stock_data['Bollinger_Middle'], stock_data['Bollinger_Lower'] = calculate_bollinger_bands(stock_data)

    stock_data.dropna(inplace=True)
    return stock_data

def normalize_data(data, feature_columns, target_column):
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    data[feature_columns] = scaler_features.fit_transform(data[feature_columns])
    data[target_column] = scaler_target.fit_transform(data[[target_column]])
    return data, scaler_features, scaler_target

def split_data(data, feature_columns, target_column):
    X = data[feature_columns].values
    y = data[target_column].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def tune_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2, 
        scoring='neg_mean_squared_error'
    )

    grid_search.fit(X_train, y_train)
    print("Best parameters for Random Forest: ", grid_search.best_params_)
    return grid_search.best_estimator_

def tune_xgboost(X_train, y_train):
    model = XGBRegressorWrapper(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [6, 10, 15],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.7, 1],
        'colsample_bytree': [0.7, 1]
    }

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2, 
        scoring='neg_mean_squared_error'
    )

    grid_search.fit(X_train, y_train.ravel())
    print("Best parameters for XGBoost: ", grid_search.best_params_)

    best_params = grid_search.best_params_
    # Create a new instance with the best parameters
    best_xgb_wrapper = XGBRegressorWrapper(**best_params)
    best_xgb_wrapper.fit(X_train, y_train.ravel())
    return best_xgb_wrapper

# Add the LSTM model
# def build_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(LSTM(50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

def build_lstm_model(input_shape):
    """
    Build an improved LSTM model for time-series prediction.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))  # First LSTM layer
    model.add(Dropout(0.2))  # Regularization
    
    model.add(LSTM(32, return_sequences=False))  # Second LSTM layer
    model.add(Dropout(0.2))

    model.add(Dense(16, activation="relu"))  # Dense layer for better mapping
    model.add(Dense(1))  # Output layer for regression

    optimizer = RMSprop(learning_rate=0.001)  # RMSprop with smaller learning rate
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # Mean Absolute Error (MAE) metric
    return model

# Prepare LSTM data
def prepare_lstm_data(data, features, target, lookback=20):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[features].iloc[i:i + lookback].values)
        y.append(data[target].iloc[i + lookback])
    return np.array(X), np.array(y)

# Ensemble Prediction Function
def ensemble_predictions(models, X):
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)

def evaluate_model(model, X_test, y_test, scaler_target):
    y_pred = model.predict(X_test)
    y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler_target.inverse_transform(y_pred.reshape(-1, 1))

    mse = mean_squared_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    
def plot_predictions(y_test, y_pred, model_name):
    """
    Plots the true vs predicted values for a given model.

    Parameters:
        y_test (array): The true values (original scale).
        y_pred (array): The predicted values (original scale).
        model_name (str): Name of the model for labeling.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--')
    plt.title(f"{model_name} - True vs Predicted")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parameters
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    features = [
        'Daily_Return', '5-Day_MA', '10-Day_MA', '5-Day_Volatility',
        'RSI', 'EMA_10', 'Bollinger_Upper', 'Bollinger_Middle',
        'Bollinger_Lower', 'Lag_1', 'Lag_2', 'Lag_5'
    ]
    target = 'Close'
    lookback = 5  # LSTM lookback window

    # Fetch and preprocess data
    print("Fetching and preprocessing data...")
    stock_data = fetch_and_preprocess_data("NVDA", start_date, end_date)
    stock_data, scaler_features, scaler_target = normalize_data(stock_data, features, target)

    # Prepare train-test splits
    X_train, X_test, y_train, y_test = split_data(stock_data, features, target)

    # Prepare LSTM data (3D for sequential input)
    print("Preparing LSTM data...")
    X_lstm, y_lstm = prepare_lstm_data(stock_data, features, target, lookback)
    X_train_lstm, X_test_lstm = X_lstm[:-20], X_lstm[-20:]
    y_train_lstm, y_test_lstm = y_lstm[:-20], y_lstm[-20:]

    # Align Random Forest and XGBoost test sets to match LSTM's test size
    X_test_aligned = X_test[-len(X_test_lstm):]
    y_test_aligned = y_test[-len(X_test_lstm):]

    # Train traditional models
    print("Tuning Random Forest...")
    best_rf_model = tune_random_forest(X_train, y_train)

    print("Tuning XGBoost...")
    best_xgb_model = tune_xgboost(X_train, y_train)

    # Train LSTM model
    print("Training LSTM model...")
    lstm_model = build_lstm_model((lookback, len(features)))
    # lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=16, validation_split=0.2, verbose=1)
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train LSTM model
    history = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,  # Increased epochs for better convergence
        batch_size=32,  # Increased batch size
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Predictions for all models
    print("Making predictions...")

    # Random Forest
    y_pred_rf = best_rf_model.predict(X_test_aligned)
    y_pred_rf_original = scaler_target.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()

    # XGBoost
    y_pred_xgb = best_xgb_model.predict(X_test_aligned)
    y_pred_xgb_original = scaler_target.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten()

    # LSTM
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
    y_pred_lstm_original = scaler_target.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

    # Ensemble: Average predictions from all models
    y_pred_ensemble = (y_pred_rf_original + y_pred_xgb_original + y_pred_lstm_original) / 3

    # Align y_test to original scale
    y_test_original = scaler_target.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()

    # Evaluation
    print("Evaluating models...")
    print("\nRandom Forest:")
    print(f"MSE: {mean_squared_error(y_test_original, y_pred_rf_original):.4f}, R²: {r2_score(y_test_original, y_pred_rf_original):.4f}")

    print("\nXGBoost:")
    print(f"MSE: {mean_squared_error(y_test_original, y_pred_xgb_original):.4f}, R²: {r2_score(y_test_original, y_pred_xgb_original):.4f}")

    print("\nLSTM:")
    print(f"MSE: {mean_squared_error(y_test_original, y_pred_lstm_original):.4f}, R²: {r2_score(y_test_original, y_pred_lstm_original):.4f}")

    print("\nEnsemble:")
    print(f"MSE: {mean_squared_error(y_test_original, y_pred_ensemble):.4f}, R²: {r2_score(y_test_original, y_pred_ensemble):.4f}")

    # Plot the results
    print("Plotting results...")
    plot_predictions(y_test_original, y_pred_rf_original, "Random Forest")
    plot_predictions(y_test_original, y_pred_xgb_original, "XGBoost")
    plot_predictions(y_test_original, y_pred_lstm_original, "LSTM")
    plot_predictions(y_test_original, y_pred_ensemble, "Ensemble Model")
    
    # Create Voting Regressor (ensemble model)
    ensemble_model = VotingRegressor(
        estimators=[
            ('rf', best_rf_model),
            ('xgb', best_xgb_model),
            ('lstm', lstm_model)  # Add LSTM as one of the models
        ]
    )

    # Save individual models
    print("Saving individual models...")
    with open("random_forest_model.pkl", "wb") as rf_file:
        pickle.dump(best_rf_model, rf_file)

    with open("xgboost_model.pkl", "wb") as xgb_file:
        pickle.dump(best_xgb_model, xgb_file)

    lstm_model.save("lstm_model.h5")  # Save LSTM model separately

    # Save the ensemble model metadata (without LSTM weights)
    print("Saving ensemble model metadata...")
    with open("ensemble_model_metadata.pkl", "wb") as ensemble_file:
        pickle.dump(ensemble_model, ensemble_file)

    print("Models saved successfully.")

    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

