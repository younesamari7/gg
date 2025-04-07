import streamlit as st

# Set page configuration (MUST be first Streamlit command)
st.set_page_config(page_title="AI Stock Forecasting Suite", layout="wide", page_icon="🚀")

# Import modular components
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

# Sidebar navigation
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to", [
    "Overview Dashboard",
    "Charts & Technical Analysis",
    "Fundamentals & Earnings",
    "News & Sentiment",
    "ML Forecast Engine",
    "DL Forecast Engine",
    "Backtesting Strategies",
    "Portfolio Tracker",
    "Alerts & Notifications",
    "Exports & Reports",
    "User Settings",
    "About / Help"
])

# App router
def app():
    if section == "Overview Dashboard":
        overview_dashboard()
    elif section == "Charts & Technical Analysis":
        charts_technical_analysis()
    elif section == "Fundamentals & Earnings":
        fundamentals_earnings()
    elif section == "News & Sentiment":
        news_sentiment()
    elif section == "ML Forecast Engine":
        ml_forecast_engine()
    elif section == "DL Forecast Engine":
        dl_forecast_engine()
    elif section == "Backtesting Strategies":
        backtesting_strategies()
    elif section == "Portfolio Tracker":
        portfolio_tracker()
    elif section == "Alerts & Notifications":
        alerts_notifications()
    elif section == "Exports & Reports":
        exports_reports()
    elif section == "User Settings":
        user_settings()
    elif section == "About / Help":
        about_help()

if __name__ == "__main__":
    app()
