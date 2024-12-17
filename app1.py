# # from flask import Flask, request, jsonify, render_template
# # import pickle
# # import numpy as np
# # import pandas as pd
# # from datetime import datetime, timedelta
# # from xgb_wrapper import XGBRegressorWrapper

# # # Initialize Flask app
# # app = Flask(__name__, template_folder="templates", static_folder="static")

# # # Load models and scalers
# # with open("model/gradient_boosting_model.pkl", "rb") as xgb_file:
# #     xgb_model = pickle.load(xgb_file)
    
# # # Simulate Open and Close Prices
# # def simulate_open_close_prices(selected_date):
# #     """ Simulates Open and Close prices for NVDA and NVDQ for 5 business days. """
# #     today = datetime.strptime(selected_date, "%Y-%m-%d")
# #     dates = []
# #     for i in range(1, 6):
# #         next_day = today + timedelta(days=i)
# #         if next_day.weekday() < 5:  # Ensure only business days (Mon-Fri)
# #             dates.append(next_day.strftime("%Y-%m-%d"))

# #     data = {
# #         "Date": dates,
# #         "Open_NVDA": np.random.uniform(150, 170, size=len(dates)),
# #         "Close_NVDA": np.random.uniform(150, 170, size=len(dates)),
# #         "Open_NVDQ": np.random.uniform(140, 160, size=len(dates)),
# #         "Close_NVDQ": np.random.uniform(140, 160, size=len(dates))
# #     }
# #     return pd.DataFrame(data)

# # @app.route("/", methods=["GET"])
# # def home():
# #     return render_template("index.html")

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # Get selected date from frontend
# #         data = request.json
# #         selected_date = data.get("date")
# #         if not selected_date:
# #             return jsonify({"error": "No date provided"}), 400

# #         # Simulate open and close prices for 5 days starting from selected date
# #         prices_df = simulate_open_close_prices(selected_date)

# #         # Decision logic for actions (BULLISH, BEARISH, IDLE)
# #         actions = []
# #         for i, row in prices_df.iterrows():
# #             if row['Close_NVDA'] > row['Open_NVDQ'] * 1.05:  # BULLISH threshold
# #                 actions.append("BULLISH")
# #             elif row['Close_NVDQ'] > row['Open_NVDA'] * 1.05:  # BEARISH threshold
# #                 actions.append("BEARISH")
# #             else:
# #                 actions.append("IDLE")

# #         # Compute final portfolio value
# #         nvda_shares = 10000
# #         nvdq_shares = 100000

# #         for i, action in enumerate(actions):
# #             if action == "BULLISH":
# #                 nvda_shares += nvdq_shares * (prices_df['Open_NVDQ'][i] / prices_df['Open_NVDA'][i])
# #                 nvdq_shares = 0
# #             elif action == "BEARISH":
# #                 nvdq_shares += nvda_shares * (prices_df['Open_NVDA'][i] / prices_df['Open_NVDQ'][i])
# #                 nvda_shares = 0

# #         final_value = nvda_shares * prices_df['Close_NVDA'].iloc[-1] + nvdq_shares * prices_df['Close_NVDQ'].iloc[-1]

# #         # Response
# #         response = {
# #             "dates": prices_df['Date'].tolist(),
# #             "open_nvda": prices_df['Open_NVDA'].tolist(),
# #             "close_nvda": prices_df['Close_NVDA'].tolist(),
# #             "open_nvdq": prices_df['Open_NVDQ'].tolist(),
# #             "close_nvdq": prices_df['Close_NVDQ'].tolist(),
# #             "actions": actions,
# #             "final_value": round(final_value, 2)
# #         }
# #         return jsonify(response)

# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # if __name__ == "__main__":
# #     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import yfinance as yf
# import logging
# from sklearn.preprocessing import MinMaxScaler
# from xgb_wrapper import XGBRegressorWrapper

# # Initialize Flask app
# app = Flask(__name__, template_folder="templates", static_folder="static")

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load pre-trained model and scaler
# with open("model/xgboost_model.pkl", "rb") as xgb_file:
#     xgb_model = pickle.load(xgb_file)
#     logging.info("Gradient Boosting model loaded successfully.")

# # Function to fetch historical data and prepare features
# def predict_prices(selected_date, model):
#     """Predict Open and Close prices for NVDA for 5 business days."""
#     today = datetime.strptime(selected_date, "%Y-%m-%d")
#     dates = []
#     features = []

#     logging.info("Fetching historical data for NVDA.")
#     # Fetch historical data for NVDA
#     start_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")  # Fetch enough data for feature generation
#     historical_data = yf.download("NVDA", start=start_date, end=selected_date)

#     if historical_data.empty:
#         logging.error("No data available for the given date range.")
#         raise ValueError("No data available for the given date range.")

#     logging.info("Generating features for predictions.")
#     # Generate features for the next 5 days
#     for i in range(1, 6):
#         next_day = today + timedelta(days=i)
#         if next_day.weekday() < 5:  # Only business days
#             dates.append(next_day.strftime("%Y-%m-%d"))

#             # Use historical data to compute features (adjust based on your model's feature set)
#             features.append([
#                 historical_data['Close'].iloc[-1],  # Last known close price
#                 historical_data['Close'].rolling(5).mean().iloc[-1],  # 5-day MA
#                 historical_data['Close'].rolling(10).mean().iloc[-1],  # 10-day MA
#                 historical_data['Close'].pct_change().iloc[-1],  # Daily return
#                 historical_data['Close'].rolling(5).std().iloc[-1]  # 5-day volatility
#             ])

#     # Convert features into a DataFrame or NumPy array (depending on your model's requirements)
#     features_array = np.array(features)

#     # Predict prices using the trained model
#     logging.info("Making predictions using the model.")
#     predictions = model.predict(features_array)

#     return pd.DataFrame({
#         "Date": dates,
#         "Predicted_Close_NVDA": predictions
#     })

# # Flask Routes
# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Log raw request data
#         logging.info(f"Request headers: {request.headers}")
#         logging.info(f"Raw request data: {request.get_data(as_text=True)}")

#         # Ensure JSON payload is received
#         if not request.is_json:
#             logging.warning("Request is not in JSON format.")
#             return jsonify({"error": "Request must be in JSON format"}), 400

#         data = request.get_json()  # Safely get JSON payload
#         selected_date = data.get("date")

#         if not selected_date:
#             logging.warning("No date provided in the request.")
#             return jsonify({"error": "No date provided"}), 400

#         logging.info(f"Received prediction request for date: {selected_date}")
        
#         # Proceed with predictions
#         predictions_df = predict_prices(selected_date, xgb_model)

#         # Dynamic thresholds and response creation...
#         mean_price = predictions_df['Predicted_Close_NVDA'].mean()
#         std_price = predictions_df['Predicted_Close_NVDA'].std()

#         bullish_threshold = mean_price + std_price
#         bearish_threshold = mean_price - std_price

#         actions = [
#             "BULLISH" if row['Predicted_Close_NVDA'] > bullish_threshold 
#             else "BEARISH" if row['Predicted_Close_NVDA'] < bearish_threshold 
#             else "IDLE"
#             for _, row in predictions_df.iterrows()
#         ]

#         response = {
#             "dates": predictions_df['Date'].tolist(),
#             "close_nvda": predictions_df['Predicted_Close_NVDA'].tolist(),
#             "actions": actions,
#             "final_value": round(10000 * predictions_df['Predicted_Close_NVDA'].iloc[-1], 2)
#         }

#         return jsonify(response)

#     except Exception as e:
#         logging.error(f"Error during prediction: {e}")
#         return jsonify({"error": str(e)}), 500


# # Run Flask app
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from xgb_wrapper import XGBRegressorWrapper
import sys
import logging

# Set up logging to output to terminal
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the trained XGBoost model
try:
    with open("model/saved_models/xgboost_model.pkl", "rb") as model_file:
        xgboost_model = pickle.load(model_file)
        logging.info("XGBoost model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise e

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
    """
    Fetch and preprocess stock data, generating the required features.
    """
    logging.info(f"Fetching data for ticker {ticker} from {start_date} to {end_date}.")
    extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    stock_data = yf.download(ticker, start=extended_start_date, end=end_date)

    if stock_data.empty:
        logging.error(f"No data found for ticker {ticker}. Check the date range or ticker symbol.")
        raise ValueError(f"No data found for ticker {ticker}. Please check the date range.")

    # Generate features
    try:
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

        stock_data.dropna(inplace=True)  # Drop rows with NaN values
        logging.info(f"Features generated successfully for ticker {ticker}.")
        return stock_data.tail(5)  # Return only the last 5 business days

    except Exception as e:
        logging.error(f"Error during feature generation: {str(e)}")
        raise e

@app.route("/")
def home():
    """
    Render the homepage with index.html.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_prices():
    """
    Endpoint to predict stock prices for the next 5 business days.
    """
    try:
        data = request.get_json()
        chosen_date = data.get("chosen_date")
        
        if not chosen_date:
            logging.warning("No chosen_date provided in the request.")
            return jsonify({"error": "chosen_date is required"}), 400

        # Fetch and preprocess stock data
        logging.info(f"Processing prediction request for chosen date: {chosen_date}")
        stock_data = fetch_and_preprocess_data(
            "NVDA",
            chosen_date,
            (datetime.strptime(chosen_date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
        )

        # Extract features
        features = stock_data[
            [
                'Daily_Return', '5-Day_MA', '10-Day_MA', '5-Day_Volatility',
                'Lag_1', 'Lag_2', 'Lag_5',
                'RSI', 'EMA_10',
                'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower'
            ]
        ].values

        logging.debug(f"Features for prediction: {features}")

        # Predict prices using the loaded model
        predictions = xgboost_model.predict(features)
        logging.info(f"Predictions: {predictions}")

        # Prepare response
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        return jsonify({"dates": dates, "predicted_prices": predictions.tolist()})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
