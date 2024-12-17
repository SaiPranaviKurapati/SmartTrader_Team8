from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder="templates", static_folder="static")

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

def predict_next_5_business_days(rf_model, recent_data, scaler_features, scaler_target, start_date, stock_data):
    predictions = []
    dates = []
    scaled_data = scaler_features.transform(recent_data.reshape(1, -1))
    current_date = datetime.strptime(start_date, "%Y-%m-%d")

    while len(predictions) < 5:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5: 
            pred = rf_model.predict(scaled_data)
            predictions.append(scaler_target.inverse_transform(pred.reshape(-1, 1)).squeeze())
            scaled_data = np.roll(scaled_data, -1, axis=1)
            scaled_data[0, -1] = pred[0]
            dates.append(current_date.strftime("%Y-%m-%d"))

    return np.max(predictions), np.min(predictions), np.mean(predictions), predictions, dates

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        start_date = data.get("start_date")
        if not start_date:
            return jsonify({"error": "Start date is required."}), 400

        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        
        stock_data = fetch_and_preprocess_data("NVDA", start_date, end_date)
        features = ['Daily_Return', '5-Day_MA', '10-Day_MA', '5-Day_Volatility']
        target = 'Close'
        stock_data, scaler_features, scaler_target = normalize_data(stock_data, features, target)

        X_train, _, y_train, _ = split_data(stock_data, features, target)
        rf_model = train_random_forest(X_train, y_train)

        recent_data = stock_data[features].iloc[-1].values
        highest, lowest, avg, predictions, dates = predict_next_5_business_days(
            rf_model, recent_data, scaler_features, scaler_target, start_date, stock_data
        )

        strategies = [("BULLISH" if pred > avg else "BEARISH") for pred in predictions]

        return jsonify({
            "highest_price": round(highest, 2),
            "lowest_price": round(lowest, 2),
            "average_price": round(avg, 2),
            "strategies": [{"date": date, "action": strategy} for date, strategy in zip(dates, strategies)]
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
