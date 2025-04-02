import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf


def portfolio_tracker():
    # --- Mock functions for AI forecasts, risk, and VaR (to be replaced with real models) ---
    def mock_forecast_prices(ticker):
        return np.random.uniform(0.05, 0.25)  # Expected return

    def mock_var(ticker, confidence=0.95):
        return np.random.uniform(0.05, 0.15)  # VaR

    def monte_carlo_simulation(ticker, days=30, trials=1000):
        start_price = np.random.uniform(50, 500)
        returns = np.random.normal(0.001, 0.02, (days, trials))
        price_paths = start_price * np.exp(np.cumsum(returns, axis=0))
        return price_paths

    def validate_symbol(symbol):
        try:
            info = yf.Ticker(symbol).info
            return 'shortName' in info
        except:
            return False

    # --- Streamlit UI ---
    st.title("🤖 AI Portfolio Recommendation System")

    st.sidebar.header("User Profile")
    capital = st.sidebar.number_input("💰 Capital to Invest ($)", min_value=1000.0, value=10000.0, step=100.0)
    risk_profile = st.sidebar.selectbox("⚖️ Risk Profile", ["Low", "Medium", "High"])
    asset_class = st.sidebar.selectbox("📊 Asset Class", ["Stocks", "Crypto", "Both"])
    time_horizon = st.sidebar.slider("⏳ Time Horizon (months)", 1, 24, 6)

    # --- Symbol Entry ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔎 Choose from common tickers (autocomplete)**")
    default_list = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "BTC-USD", "ETH-USD", "SOL-USD"]
    selected_autocomplete = st.sidebar.multiselect("Select assets:", options=default_list, default=["AAPL", "BTC-USD"])

    st.sidebar.markdown("**📝 Or enter custom tickers (comma-separated):**")
    user_input = st.sidebar.text_area("", value="AAPL, MSFT, BTC-USD, ETH-USD")
    user_symbols = [sym.strip().upper() for sym in user_input.split(",") if sym.strip()] + selected_autocomplete

    # --- Validate Symbols ---
    valid_symbols = []
    invalid_symbols = []
    for sym in user_symbols:
        if validate_symbol(sym):
            valid_symbols.append(sym)
        else:
            invalid_symbols.append(sym)

    if invalid_symbols:
        st.warning(f"⚠️ Invalid symbols skipped: {', '.join(invalid_symbols)}")

    def determine_type(symbol):
        return "Crypto" if "-USD" in symbol or symbol in ["BTC", "ETH", "SOL", "ADA"] else "Stock"

    # --- AI Recommendation Engine ---
    recommendations = []
    total_weight = 0
    for ticker in valid_symbols:
        exp_return = mock_forecast_prices(ticker)
        var_value = mock_var(ticker)
        risk_multiplier = {"Low": 0.5, "Medium": 1, "High": 1.5}[risk_profile]
        score = exp_return / (var_value * risk_multiplier)
        weight = max(score, 0.01)
        total_weight += weight
        recommendations.append({
            "Asset": ticker,
            "Type": determine_type(ticker),
            "Expected Return": round(exp_return * 100, 2),
            "VaR (95%)": round(var_value * 100, 2),
            "Score": round(score, 2),
            "Weight": weight
        })

    # --- Normalize Weights and Calculate Allocation ---
    for rec in recommendations:
        rec["Allocation %"] = round((rec["Weight"] / total_weight) * 100, 2)
        rec["Amount to Invest"] = round((rec["Allocation %"] / 100) * capital, 2)

    # --- Display Table ---
    df = pd.DataFrame(recommendations).sort_values(by="Score", ascending=False)
    if not df.empty:
        st.subheader("📈 Recommended Portfolio")
        st.dataframe(df[["Asset", "Type", "Expected Return", "VaR (95%)", "Allocation %", "Amount to Invest"]], use_container_width=True)

        # --- Monte Carlo Sim ---
        st.subheader("🎲 Monte Carlo Simulation (1 Asset)")
        sim_asset = st.selectbox("Choose asset to simulate:", df["Asset"])
        sim_result = monte_carlo_simulation(sim_asset)

        fig, ax = plt.subplots()
        ax.plot(sim_result[:, :10])  # Show first 10 trials
        ax.set_title(f"Monte Carlo Price Simulation: {sim_asset}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        # --- Export ---
        st.download_button("📥 Download Portfolio as CSV", data=df.to_csv(index=False), file_name="ai_portfolio.csv")
    else:
        st.info("💡 Please enter at least one valid symbol to generate recommendations.")