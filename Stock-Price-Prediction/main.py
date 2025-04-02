import streamlit as st

# ✅ Only ONE set_page_config, immediately after importing Streamlit
st.set_page_config(page_title="AI Stock Forecasting Suite", layout="wide")

# ✅ Import other modules AFTER set_page_config
from global_sidebar import global_sidebar
from overview_dashboard import overview_dashboard
from charts_technical_analysis import charts_technical_analysis
from fundamentals_earnings import fundamentals_earnings
from news_sentiment import news_sentiment
from ml_engine import ml_forecast_engine
from dl_engine import dl_forecast_engine
from backtesting_strategies import backtesting_strategies
from portfolio_tracker import portfolio_tracker
from alerts_notifications import alerts_notifications
from exports_reports import exports_reports
from user_settings import user_settings
from about_help import about_help

def main():
    selected_page = global_sidebar()

    if selected_page == "Overview Dashboard":
        overview_dashboard()
    elif selected_page == "Charts & Technical Analysis":
        charts_technical_analysis()
    elif selected_page == "Fundamentals & Earnings":
        fundamentals_earnings()
    elif selected_page == "News & Sentiment":
        news_sentiment()
    elif selected_page == "ML Forecast Engine":
        ml_forecast_engine()
    elif selected_page == "DL Forecast Engine":
        dl_forecast_engine()
    elif selected_page == "Backtesting & Strategies":
        backtesting_strategies()
    elif selected_page == "Portfolio Tracker":
        portfolio_tracker()
    elif selected_page == "Alerts & Notifications":
        alerts_notifications()
    elif selected_page == "Exports & Reports":
        exports_reports()
    elif selected_page == "User Settings":
        user_settings()
    elif selected_page == "About & Help":
        about_help()

if __name__ == "__main__":
    main()
