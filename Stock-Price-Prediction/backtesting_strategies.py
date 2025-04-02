import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# === RSI Calculation Function ===
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === Backtesting App ===
def backtesting_strategies():
    st.title("📈 Backtesting & Strategies")
    st.subheader("📊 Simple Moving Average (SMA) Crossover with RSI Strategy")

    # === User Inputs ===
    ticker = st.text_input("Symbol:", st.session_state.get("ticker", "AAPL"))
    st.session_state.ticker = ticker.upper()

    short_window = st.number_input("Short Window (days)", min_value=5, max_value=50, value=10)
    long_window = st.number_input("Long Window (days)", min_value=20, max_value=200, value=50)
    rsi_threshold = st.slider("RSI Filter Threshold", min_value=30, max_value=70, value=50)

    try:
        df = yf.download(ticker, period="1y")
        df = df.dropna()

        # === Calculate Indicators ===
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        df['RSI'] = calculate_rsi(df['Close'])

        # === Strategy Signal ===
        df['Signal'] = 0
        df.loc[(df['SMA_Short'] > df['SMA_Long']) & (df['RSI'] > rsi_threshold), 'Signal'] = 1
        df.loc[(df['SMA_Short'] < df['SMA_Long']) & (df['RSI'] < (100 - rsi_threshold)), 'Signal'] = -1

        # === Strategy Return Calculation ===
        df['Daily_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
        df['Equity_Curve'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()

        # === Plot Equity Curve ===
        st.subheader("📉 Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Curve'], mode='lines', name='Equity Curve'))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # === Trade Log ===
        st.subheader("📋 Trade Log")
        df['Trade'] = df['Signal'].diff().fillna(0)
        trades = df[df['Trade'] != 0][['Close', 'RSI', 'SMA_Short', 'SMA_Long', 'Signal']]
        st.dataframe(trades.tail(10))

        # === Explanation ===
        with st.expander("📘 Strategy Explanation"):
            st.markdown("""
                - This strategy combines **Simple Moving Averages (SMA)** with **Relative Strength Index (RSI)**.
                - A **Buy Signal** is generated when the short SMA crosses above the long SMA **and** RSI is above the threshold.
                - A **Sell Signal** occurs when the short SMA drops below the long SMA **and** RSI is below the inverse threshold (e.g., 100 - 50 = 50).
                - The **Equity Curve** simulates cumulative returns assuming the strategy was followed daily.
            """)

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")

if __name__ == '__main__':
    backtesting_strategies()