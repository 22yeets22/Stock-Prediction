import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os.path import exists
from constants import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(file_path):
    """Load and preprocess the test data."""
    if not exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    test_data = pd.read_csv(file_path)
    required_columns = ['Open', 'Close', 'Volume', 'High', 'Low', 'Adj Close']
    
    if not all(col in test_data.columns for col in required_columns):
        raise ValueError(f"The CSV file must contain all of these columns: {required_columns}")
    
    scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in required_columns}
    scaled_data = {col: scalers[col].fit_transform(test_data[col].values.reshape(-1, 1)) 
                   for col in required_columns}
    
    return test_data, scaled_data, scalers['Close']


def prepare_input_data(scaled_data):
    """Prepare input data for the model."""
    x_test = []
    for i in range(prediction_days_past, len(scaled_data['Open'])):
        x_test.append([scaled_data[col][i-prediction_days_past:i, 0] for col in scaled_data.keys()])
    
    x_test = np.array(x_test)
    return np.reshape(x_test, (x_test.shape[0], prediction_days_past, len(scaled_data)))


def plot_predictions(actual_prices, predicted_prices, title):
    """Plot actual vs predicted prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color="dodgerblue", label="Actual Prices")
    plt.plot(range(prediction_days_past, len(predicted_prices) + prediction_days_past),
             predicted_prices, color="lightcoral", label="Predicted Prices")
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Share Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Load the model
    try:
        print("Loading model...")
        model = load_model(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Load and preprocess the test data
    try:
        test_data, scaled_data, close_scaler = load_and_preprocess_data(f"data/test/{stock}.csv")
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        return

    # Prepare input data
    x_test = prepare_input_data(scaled_data)

    # Make predictions
    try:
        predicted_prices = model.predict(x_test, verbose=0)
        predicted_prices = close_scaler.inverse_transform(predicted_prices)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return

    # Plot the results
    actual_close = test_data['Close'].values
    plot_predictions(actual_close, predicted_prices, f"{stock} Share Prices")


if __name__ == "__main__":
    stock = "NVDA"
    main()
