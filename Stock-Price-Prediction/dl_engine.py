import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping


def dl_forecast_engine():
    st.title("\U0001F680 Deep Learning Forecast Engine | LSTM + GRU + RNN")

    asset_type = st.sidebar.selectbox("Select Asset Type", ["Stock", "Crypto"])
    ticker = st.sidebar.text_input("Enter Ticker", value="AAPL" if asset_type == "Stock" else "BTC-USD")

    lookback = st.sidebar.slider("Lookback Window (Days)", 30, 120, 60)
    model_type = st.sidebar.selectbox("Model Type", ["LSTM", "GRU", "Bidirectional LSTM", "Simple RNN", "Hybrid LSTM+GRU"])

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

    df = yf.download(ticker, period="5y")
    if df.empty:
        st.error(f"No data found for ticker {ticker}. Please check the ticker symbol and try again.")
        return

    data = df[['Close']].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data) - forecast_days):
        x_train.append(scaled_data[i - lookback:i, 0])
        y_train.append(scaled_data[i:i + forecast_days, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(64, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    elif model_type == "GRU":
        model.add(GRU(64, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(x_train.shape[1], 1)))
    elif model_type == "Simple RNN":
        model.add(SimpleRNN(64, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    elif model_type == "Hybrid LSTM+GRU":
        model.add(LSTM(32, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(GRU(32))

    model.add(Dropout(0.2))
    model.add(Dense(forecast_days))
    model.compile(optimizer='adam', loss='mse')

    with st.spinner("Training deep learning model..."):
        model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=3)])

    last_sequence = scaled_data[-lookback:]
    last_sequence = np.reshape(last_sequence, (1, lookback, 1))
    forecast_scaled = model.predict(last_sequence)[0]
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    start_date = df.index[-1] + timedelta(days=1) if forecast_start is None else pd.to_datetime(forecast_start)
    forecast_dates = pd.date_range(start=start_date, periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})

    # Evaluate on last known values (optional testing window)
    y_true_scaled = scaled_data[-forecast_days:]
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, forecast)
    mae = mean_absolute_error(y_true, forecast)
    r2 = r2_score(y_true, forecast)
    direction_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(forecast))) * 100

    st.subheader("\U0001F4C9 Performance Metrics")
    st.markdown(f"- **MSE:** {mse:.4f}")
    st.markdown(f"- **MAE:** {mae:.4f}")
    st.markdown(f"- **R² Score:** {r2:.4f}")
    st.markdown(f"- **Directional Accuracy:** {direction_acc:.2f}%")

    # Plot
    st.subheader("\U0001F4C8 Forecast Chart")
    
    # Create figure with constrained layout
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    # Plot historical data with simple line
    ax.plot(df.index[-100:], df['Close'].iloc[-100:], 
            label="Historical", 
            color='#3498db', 
            linewidth=2)
    
    # Plot forecast with simple line
    ax.plot(forecast_df['Date'], forecast_df['Predicted Price'], 
            label=f"{model_type} Forecast", 
            color='#e74c3c', 
            linewidth=2)
    
    # Modern title and labels
    ax.set_title(f"{ticker} Price Forecast | {model_type} Model", 
                fontsize=16, 
                fontweight='bold',
                pad=25)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Price (USD)", fontsize=13)
    
    # Improved legend
    ax.legend(frameon=True, 
             facecolor='white', 
             framealpha=0.95, 
             fontsize=12,
             loc='upper left',
             borderpad=1)
    
    # Grid and ticks
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Remove spines and set background
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('#f5f5f5')
    
    # Add watermark with model info
    fig.text(0.88, 0.15, 
            f"{model_type} Model\n{lookback}D Lookback\n{forecast_days}D Forecast",
            fontsize=11,
            color='#95a5a6',
            alpha=0.4,
            ha='center',
            va='center',
            rotation=30)
    
    # Add subtle border
    fig.patch.set_edgecolor('#e0e0e0')
    fig.patch.set_linewidth(1)

    st.pyplot(fig)

    st.subheader("\U0001F4CA Forecast Table")
    st.dataframe(forecast_df)
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast CSV", data=csv, file_name=f"{ticker}_{model_type.lower()}_forecast.csv", mime="text/csv")


if __name__ == '__main__':
    dl_forecast_engine()
