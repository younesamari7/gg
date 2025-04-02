# alerts_notifications.py
import streamlit as st
import yfinance as yf
from datetime import datetime

def alerts_notifications():
    st.title("Alerts & Notifications")
    ticker = st.session_state.ticker
    asset = yf.Ticker(ticker)

    st.subheader("Set Thresholds")
    price_threshold = st.number_input("Price Threshold", value=100.0)
    rsi_threshold = st.number_input("RSI Threshold", value=30.0)
    macd_alert = st.checkbox("MACD Crossover Alert", value=False)

    try:
        data = asset.history(period="1d")
        current_price = data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return

    st.write(f"Current Price: ${current_price:.2f}")
    alerts_triggered = []
    if current_price > price_threshold:
        alerts_triggered.append("Price exceeded threshold")
    # Note: Calculating RSI and MACD would require additional code.
    if rsi_threshold:
        alerts_triggered.append("RSI alert not implemented (requires technical indicator computation)")
    if macd_alert:
        alerts_triggered.append("MACD alert not implemented (requires technical indicator computation)")

    st.subheader("Alert History Log")
    if alerts_triggered:
        for alert in alerts_triggered:
            st.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {alert}")
    else:
        st.write("No alerts triggered.")
