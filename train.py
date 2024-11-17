import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Constants (assuming these are defined in a separate file)
from constants import prediction_days_past, prediction_days_future, model_name

# Load data from multiple CSV files
data_files = ['AAPL.csv', 'BRK-A.csv', 'GOOGL.csv', 'MSFT.csv']
datasets = [pd.read_csv(f'data/train/{file}') for file in data_files]


# Prepare data
def prep_data(data):
    scalers = {
        'Open': MinMaxScaler(feature_range=(0, 1)),
        'Close': MinMaxScaler(feature_range=(0, 1)),
        'Volume': MinMaxScaler(feature_range=(0, 1)),
        'High': MinMaxScaler(feature_range=(0, 1)),
        'Low': MinMaxScaler(feature_range=(0, 1)),
        'Adj Close': MinMaxScaler(feature_range=(0, 1))
    }
    
    scaled_data = {col: scalers[col].fit_transform(data[col].values.reshape(-1, 1)) 
                   for col in scalers.keys()}

    x_train, y_train = [], []

    for x in range(prediction_days_past, len(scaled_data['Open']) - prediction_days_future + 1):
        x_train.append([scaled_data[col][x-prediction_days_past:x, 0] for col in scaled_data.keys()])
        y_train.append([scaled_data['Close'][x + day, 0] for day in range(prediction_days_future)])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], prediction_days_past, len(scaled_data)))
    return x_train, y_train, scalers['Close']


# Build model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=60, return_sequences=True, input_shape=input_shape),
        Dropout(0.25),
        LSTM(units=50, return_sequences=True),
        Dropout(0.25),
        LSTM(units=40, return_sequences=True),
        Dropout(0.25),
        LSTM(units=30),
        Dropout(0.25),
        Dense(units=prediction_days_future)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Prepare datasets
all_x_train, all_y_train = [], []
for data in datasets:
    x_train, y_train, _ = prep_data(data)
    all_x_train.append(x_train)
    all_y_train.append(y_train)

# Concatenate all datasets
x_train = np.concatenate(all_x_train)
y_train = np.concatenate(all_y_train)

# Build and train model
model = build_model((x_train.shape[1], x_train.shape[2]))

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(
    x_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stop, reduce_lr]
)

# Save the model
model.save(model_name)
