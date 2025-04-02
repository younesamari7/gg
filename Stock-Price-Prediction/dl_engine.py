import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

def dl_forecast_engine():
    st.title("🚀 Deep Learning Forecast Engine | LSTM + GRU")
    
    # Let the user choose the asset type and enter a ticker
    asset_type = st.sidebar.selectbox("Select Asset Type", ["Stock", "Crypto"])
    if asset_type == "Stock":
        ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    else:
        ticker = st.sidebar.text_input("Enter Crypto Ticker", value="BTC-USD")
        
    lookback = st.sidebar.slider("Lookback Window (Days)", 30, 120, 60)
    model_type = st.sidebar.selectbox("Model Type", ["LSTM", "GRU", "Bidirectional LSTM"])

    forecast_option = st.sidebar.radio("Forecast Type", ["Next N Days", "Forecast Until Date", "Date Range"])

    if forecast_option == "Next N Days":
        forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 5, 60, 15)
        forecast_start = None
        forecast_end = None
    elif forecast_option == "Forecast Until Date":
        forecast_end = st.sidebar.date_input("Select Forecast End Date", value=datetime.today() + timedelta(days=15))
        forecast_days = (forecast_end - datetime.today().date()).days
        forecast_start = None
    elif forecast_option == "Date Range":
        forecast_start = st.sidebar.date_input("Start Date", value=datetime.today())
        forecast_end = st.sidebar.date_input("End Date", value=datetime.today() + timedelta(days=15))
        forecast_days = (forecast_end - forecast_start).days

    st.markdown(f"Fetching and modeling **{ticker}** using {model_type} neural network...")

    # Load and scale data
    df = yf.download(ticker, period="5y")
    if df.empty:
        st.error(f"No data found for ticker {ticker}. Please check the ticker symbol and try again.")
        return
    data = df[['Close']].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Prepare training data
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data) - forecast_days):
        x_train.append(scaled_data[i - lookback:i, 0])
        y_train.append(scaled_data[i:i + forecast_days, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the selected model
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(64, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    elif model_type == "GRU":
        model.add(GRU(64, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(x_train.shape[1], 1)))

    model.add(Dropout(0.2))
    model.add(Dense(forecast_days))
    model.compile(optimizer='adam', loss='mse')

    with st.spinner("Training deep learning model..."):
        model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=3)])

    # Forecast future prices
    last_sequence = scaled_data[-lookback:]
    last_sequence = np.reshape(last_sequence, (1, lookback, 1))
    forecast_scaled = model.predict(last_sequence)[0]
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    start_date = df.index[-1] + timedelta(days=1) if forecast_start is None else pd.to_datetime(forecast_start)
    forecast_dates = pd.date_range(start=start_date, periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})

    # Plot the forecast
    st.subheader("📈 Forecast Chart")
    fig, ax = plt.subplots()
    ax.plot(df.index[-200:], df['Close'].iloc[-200:], label="Historical")
    ax.plot(forecast_df['Date'], forecast_df['Predicted Price'], label=model_type, linestyle='dashed')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Output the forecast table and provide a CSV download option
    st.subheader("📊 Forecast Table")
    st.dataframe(forecast_df)
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast CSV", data=csv, file_name=f"{ticker}_{model_type.lower()}_forecast.csv", mime="text/csv")

if __name__ == '__main__':
    dl_forecast_engine()
