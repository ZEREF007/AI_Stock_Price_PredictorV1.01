Stock Price Predictor

Overview:
Predicts stock prices using an LSTM model trained on historical data. Includes an interactive UI for visualization.

Features:
- Fetches historical stock data via yFinance API
- Preprocesses data with MinMaxScaler
- LSTM model with 2 LSTM layers and 2 Dense layers
- Trains with batch size 25 over 100 epochs
- Visualizes training/validation loss and actual vs predicted prices
- Interactive UI using Streamlit

Model Performance:
- Achieved RMSE of 2.74

Tech Stack:
- Python
- Keras (TensorFlow), NumPy, Pandas, MinMaxScaler, Matplotlib, Seaborn, yFinance, Streamlit

Setup:
1. Clone repo: git clone https://github.com/ZEREF007/stock-price-predictor.git
2. Create & activate virtual environment:
   - python -m venv venv
   - source venv/bin/activate (or venv\Scripts\activate on Windows)
3. Install packages: pip install -r requirements.txt
4. Train model: python stock_price_prediction.py
5. Start UI: streamlit run stock_price_predictor_app.py

Usage:
- Train the model with stock_price_prediction.py
- Use Streamlit UI with stock_price_predictor_app.py to input stock symbols and visualize predictions

License:
MIT License

Contact:
For questions or suggestions, open an issue or contact your email.