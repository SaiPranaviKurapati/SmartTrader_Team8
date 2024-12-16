import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import matplotlib.pyplot as plt


class StockPredictor:
    def __init__(self):
        self.model = Ridge(alpha=1.0)  
        self.features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_20', 'Volatility', 'Price_Change', 'Volume_Change'
        ]

    def preprocess_data(self, df):
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()

    def prepare_features(self, df):
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        return df

    def prepare_targets(self, df):
        df['Highest_5d'] = df['High'].rolling(window=5).max().shift(-5)
        df['Lowest_5d'] = df['Low'].rolling(window=5).min().shift(-5)
        df['Avg_Close_5d'] = df['Close'].rolling(window=5).mean().shift(-5)
        return df

    def train(self, data_path, model_path):
        df = pd.read_csv('/Users/himaswetha/Desktop/257/data/raw/nvda_data.csv')

        df = self.preprocess_data(df)

        df = self.prepare_features(df)
        df = self.prepare_targets(df)

        df = df.dropna()

        X = df[self.features]
        y = df[['Highest_5d', 'Lowest_5d', 'Avg_Close_5d']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }

        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')

        print(f"Model trained and saved to {model_path}")
        print(f"Regression Metrics: {metrics}")
        print(f"Cross-Validation R² Scores: {cv_scores}")
        print(f"Mean Cross-Validation R²: {cv_scores.mean()}")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    model = StockPredictor()
    data_path = "/Users/himaswetha/Desktop/257/data/raw/nvda_data.csv"  
    model_path = "models/stock_predictor_model.pkl"  
    model.train(data_path, model_path)