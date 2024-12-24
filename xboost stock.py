import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Fetch data for TCS stock from Yahoo Finance
tcs_data = yf.download("TITAN.NS", start="2015-01-01", end="2045-01-01")

# Step 2: Preprocess the data
# Use only 'Open', 'High', 'Low', 'Close', 'Volume' columns
data = tcs_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Create additional features (e.g., rolling averages, percentage change)
data['Rolling_Mean'] = data['Close'].rolling(window=10).mean()
data['Price_Change'] = data['Close'].pct_change()

# Drop missing values
data.dropna(inplace=True)

# Prepare features (X) and target (y)
X = data[['Open', 'High', 'Low', 'Volume', 'Rolling_Mean', 'Price_Change']]
y = data['Close']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Step 7: Visualize the actual vs predicted stock prices
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red')
plt.legend()
plt.title("TCS Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.show()

# Step 8: Predict the stock price for a specific date
def predict_for_date(date):
    # Check if the input date is in the dataset
    if date not in tcs_data.index:
        print(f"Date {date} is not available in the dataset.")
        return None
    
    # Prepare the data for the specific date
    date_data = tcs_data.loc[date]
    date_features = pd.DataFrame({
        'Open': [date_data['Open']],
        'High': [date_data['High']],
        'Low': [date_data['Low']],
        'Volume': [date_data['Volume']],
        'Rolling_Mean': [data['Rolling_Mean'].loc[date].mean()],
        'Price_Change': [data['Price_Change'].loc[date].mean()]
    })
    
    # Make prediction using the trained model
    predicted_price = model.predict(date_features)
    
    return predicted_price[0]

# Input date in 'YYYY-MM-DD' format
input_date = '2024-01-01'  # Change this to any valid date from the dataset
predicted_price = predict_for_date(input_date)

if predicted_price is not None:
    print(f"Predicted stock price for {input_date}: {predicted_price}")
