import streamlit as st

def global_sidebar():
    st.sidebar.title("📊 Navigation")
    options = [
        "Overview Dashboard",
        "Charts & Technical Analysis",
        "Fundamentals & Earnings",
        "News & Sentiment",
        "ML Forecast Engine",
        "DL Forecast Engine",
        "Backtesting & Strategies",
        "Portfolio Tracker",
        "Alerts & Notifications",
        "Exports & Reports",
        "User Settings",
        "About & Help"
    ]
    return st.sidebar.radio("Select a page:", options)