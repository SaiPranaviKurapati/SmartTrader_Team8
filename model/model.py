import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

def fetch_and_preprocess_data(ticker, start_date, end_date):
    extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    stock_data = yf.download(ticker, start=extended_start_date, end=end_date)

    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}. Please check the date range.")

    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['5-Day_MA'] = stock_data['Close'].rolling(5).mean()
    stock_data['10-Day_MA'] = stock_data['Close'].rolling(10).mean()
    stock_data['5-Day_Volatility'] = stock_data['Close'].rolling(5).std()
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

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15)
    rf.fit(X_train, y_train)
    return rf

if __name__ == "__main__":
    # Parameters
    start_date = "2023-01-01"
    end_date = "2024-12-01"
    features = ['Daily_Return', '5-Day_MA', '10-Day_MA', '5-Day_Volatility']
    target = 'Close'

    # Fetch and preprocess data
    stock_data = fetch_and_preprocess_data("NVDA", start_date, end_date)
    stock_data, scaler_features, scaler_target = normalize_data(stock_data, features, target)

    # Train model
    X_train, _, y_train, _ = split_data(stock_data, features, target)
    rf_model = train_random_forest(X_train, y_train)

    # Save model and scalers
    with open("random_forest_model.pkl", "wb") as model_file:
        pickle.dump(rf_model, model_file)
    with open("scaler_X.pkl", "wb") as scaler_X_file:
        pickle.dump(scaler_features, scaler_X_file)
    with open("scaler_y.pkl", "wb") as scaler_y_file:
        pickle.dump(scaler_target, scaler_y_file)

    print("Model training complete. Files saved.")
