import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import seaborn as sns
from keras.models import load_model

# Streamlit app title
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Enhanced Stock Price Predictor App")
st.markdown("This app predicts stock prices using a pre-trained LSTM model and displays various insights.")

# Inject custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Input for stock symbol
stock = st.text_input("Enter the Stock Symbol (e.g., AAPL)", "AAPL")

# Fetch stock data
st.info("Downloading stock data...")
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
try:
    stock_data = yf.download(stock, start, end)
    st.success("Data downloaded!")
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# Display stock data
st.subheader("Stock Data")
st.dataframe(stock_data, use_container_width=True)

# Load the pre-trained model
try:
    model = load_model("Latest_stock_price_model.keras")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# User-defined moving average period
ma_period = st.slider("Select Moving Average Period", min_value=50, max_value=365, value=200)
stock_data[f'MA_{ma_period}d'] = stock_data['Close'].rolling(ma_period).mean()

# Plot Moving Averages
st.subheader(f"Close Price and MA for {ma_period} days")
fig, ax = plt.subplots(figsize=(14, 7))
sns.lineplot(x=stock_data.index, y=stock_data['Close'], label='Close Price', ax=ax, color='royalblue')
sns.lineplot(x=stock_data.index, y=stock_data[f'MA_{ma_period}d'], label=f'MA {ma_period} days', ax=ax, color='orange', alpha=0.7)
ax.set_title(f"Close Price and Moving Average ({ma_period} days)", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price", fontsize=12)
ax.legend(loc='upper left')
ax.grid(True)
st.pyplot(fig)

# Prepare data for prediction
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[['Close']].fillna(method='bfill'))

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
try:
    predictions = model.predict(x_data)
    st.success("Predictions made successfully!")
except Exception as e:
    st.error(f"Error making predictions: {e}")
    st.warning("Prediction might not be available. Please check your model or data.")

# Inverse transform and prepare data for plotting
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)
plotting_data = pd.DataFrame({
    'Original': inv_y_test.reshape(-1),
    'Predicted': inv_predictions.reshape(-1)
}, index=stock_data.index[len(stock_data) - len(predictions):])

# Display predictions
st.subheader("Original vs Predicted Values")
st.line_chart(plotting_data)

# Plot original vs predicted values
st.subheader('Close Price vs Predicted')
fig, ax = plt.subplots(figsize=(14, 7))
sns.lineplot(x=stock_data.index, y=stock_data['Close'], label='Close Price', ax=ax, color='royalblue')
sns.lineplot(x=plotting_data.index, y=plotting_data['Predicted'], label='Predicted', ax=ax, linestyle='dashed', color='red', alpha=0.7)
ax.set_title("Close Price vs Predicted", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price", fontsize=12)
ax.legend(loc='upper left')
ax.grid(True)
st.pyplot(fig)

# Download plot button (placeholder)
def download_plot():
    st.write("Download Plot functionality is not yet implemented. Please use other means to save the plot.")

st.button("Download Plot", on_click=download_plot)