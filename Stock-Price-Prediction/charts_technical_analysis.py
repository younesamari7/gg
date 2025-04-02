import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def charts_technical_analysis():
    st.title("Advanced Charts & Technical Analysis")

    # Sidebar configuration for ticker and data range
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL", key="ticker_input")
    data_range_choice = st.sidebar.radio("Data Range Option", ("Last N Days", "Custom Date Range"), key="data_range_option")
    
    if data_range_choice == "Last N Days":
        days = st.sidebar.number_input("Number of Days", min_value=30, value=120, step=10, key="days_input")
        try:
            df = yf.download(ticker, period=f"{days}d")
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return
    else:
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"), key="start_date")
        end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"), key="end_date")
        if start_date >= end_date:
            st.error("Error: Start date must be before end date.")
            return
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return

    if df.empty:
        st.error("No data found for the given ticker/date range. Please check your inputs.")
        return

    # Sidebar for chart type and technical indicators
    chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "OHLC", "Line"], key="chart_type")
    
    st.sidebar.subheader("Technical Indicators")
    add_sma = st.sidebar.checkbox("Simple Moving Average (SMA)", value=True, key="add_sma")
    sma_period = st.sidebar.number_input("SMA Period", min_value=1, value=20, step=1, key="sma_period") if add_sma else 20

    add_ema = st.sidebar.checkbox("Exponential Moving Average (EMA)", value=True, key="add_ema")
    ema_period = st.sidebar.number_input("EMA Period", min_value=1, value=20, step=1, key="ema_period") if add_ema else 20

    add_bbands = st.sidebar.checkbox("Bollinger Bands", value=True, key="add_bbands")
    bb_period = st.sidebar.number_input("BB Period", min_value=1, value=20, step=1, key="bb_period") if add_bbands else 20
    bb_std = st.sidebar.number_input("BB Std Dev", min_value=0.1, value=2.0, step=0.1, key="bb_std") if add_bbands else 2.0

    add_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=True, key="add_rsi")
    rsi_period = st.sidebar.number_input("RSI Period", min_value=1, value=14, step=1, key="rsi_period") if add_rsi else 14

    add_macd = st.sidebar.checkbox("MACD", value=True, key="add_macd")
    macd_fast = 12  # Standard MACD fast period
    macd_slow = 26  # Standard MACD slow period
    macd_signal = 9  # Standard MACD signal period

    add_obv = st.sidebar.checkbox("On Balance Volume (OBV)", value=False, key="add_obv")

    add_atr = st.sidebar.checkbox("Average True Range (ATR)", value=False, key="add_atr")
    atr_period = st.sidebar.number_input("ATR Period", min_value=1, value=14, step=1, key="atr_period") if add_atr else 14

    st.subheader(f"Interactive Price Chart: {ticker}")

    # Calculate technical indicators
    if add_sma:
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()
    if add_ema:
        df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    if add_bbands:
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        df['BB_Std'] = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + bb_std * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - bb_std * df['BB_Std']
    if add_rsi:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    if add_macd:
        df['EMA_fast'] = df['Close'].ewm(span=macd_fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=macd_slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_Signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    if add_obv:
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    if add_atr:
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=atr_period).mean()

    # Define indicator order for subplots (in fixed order)
    indicator_order = []
    if add_rsi:
        indicator_order.append("RSI")
    if add_macd:
        indicator_order.append("MACD")
    if add_obv:
        indicator_order.append("OBV")
    if add_atr:
        indicator_order.append("ATR")
    n_indicators = len(indicator_order)

    # Define subplot layout: main chart plus one row per indicator
    if n_indicators > 0:
        main_height = 0.5
        other_height = (1 - main_height) / n_indicators
        row_heights = [main_height] + [other_height] * n_indicators
    else:
        row_heights = [1.0]

    specs = [[{"secondary_y": True}]]  # Main chart with secondary y for volume
    for _ in range(n_indicators):
        specs.append([{"secondary_y": False}])

    fig = make_subplots(rows=1+n_indicators, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights, specs=specs)
    
    current_row = 1
    # Plot main price chart
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name="Price"), row=current_row, col=1)
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(x=df.index,
                              open=df['Open'],
                              high=df['High'],
                              low=df['Low'],
                              close=df['Close'],
                              name="Price"), row=current_row, col=1)
    else:  # Line chart
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode="lines", name="Close Price"), row=current_row, col=1)

    # Overlay technical indicators on main chart
    if add_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode="lines", name=f"SMA ({sma_period})"), row=current_row, col=1)
    if add_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode="lines", name=f"EMA ({ema_period})"), row=current_row, col=1)
    if add_bbands:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode="lines", line=dict(dash="dash"), name="BB Upper"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode="lines", line=dict(dash="dot"), name="BB Middle"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode="lines", line=dict(dash="dash"), name="BB Lower"), row=current_row, col=1)

    # Add volume bars on a secondary y-axis for main chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='lightgray', name="Volume", opacity=0.3),
                  row=current_row, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price", row=current_row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=current_row, col=1, secondary_y=True)

    # Plot additional indicator subplots
    for indicator in indicator_order:
        current_row += 1
        if indicator == "RSI":
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode="lines", name="RSI"), row=current_row, col=1)
            # Overbought/Oversold lines
            fig.add_hline(y=70, line=dict(dash="dash", color="red"), row=current_row, col=1)
            fig.add_hline(y=30, line=dict(dash="dash", color="green"), row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])
        elif indicator == "MACD":
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode="lines", name="MACD"), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode="lines", name="MACD Signal"), row=current_row, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD Histogram"), row=current_row, col=1)
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        elif indicator == "OBV":
            fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode="lines", name="OBV"), row=current_row, col=1)
            fig.update_yaxes(title_text="OBV", row=current_row, col=1)
        elif indicator == "ATR":
            fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode="lines", name="ATR"), row=current_row, col=1)
            fig.update_yaxes(title_text="ATR", row=current_row, col=1)

    fig.update_layout(title=f"{ticker} Price Chart with Advanced Technical Analysis", xaxis_title="Date", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Data download button
    csv = df.to_csv().encode('utf-8')
    st.download_button("Download Data as CSV", csv, "data.csv", "text/csv", key="download_csv")

    # Summary table of latest indicator values
    summary = {"Price": df['Close'].iloc[-1]}
    if add_sma:
        summary[f"SMA ({sma_period})"] = round(df['SMA'].iloc[-1], 2)
    if add_ema:
        summary[f"EMA ({ema_period})"] = round(df['EMA'].iloc[-1], 2)
    if add_bbands:
        summary["BB Upper"] = round(df['BB_Upper'].iloc[-1], 2)
        summary["BB Middle"] = round(df['BB_Middle'].iloc[-1], 2)
        summary["BB Lower"] = round(df['BB_Lower'].iloc[-1], 2)
    if add_rsi:
        summary["RSI"] = round(df['RSI'].iloc[-1], 2)
    if add_macd:
        summary["MACD"] = round(df['MACD'].iloc[-1], 2)
        summary["MACD Signal"] = round(df['MACD_Signal'].iloc[-1], 2)
    if add_obv:
        summary["OBV"] = int(df['OBV'].iloc[-1])
    if add_atr:
        summary["ATR"] = round(df['ATR'].iloc[-1], 2)
        
    st.subheader("Latest Indicator Values")
    st.dataframe(pd.DataFrame(summary, index=[0]).T)

if __name__ == '__main__':
    charts_technical_analysis()
