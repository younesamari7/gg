import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

def fetch_data(ticker, period="5y"):
    df = yf.download(ticker, period=period)[['Close']].dropna().reset_index()
    df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    return df

def forecast_prices(X, y, model, future_scaled):
    model.fit(X, y)
    preds = model.predict(future_scaled)
    y_pred_hist = model.predict(X)
    mse = mean_squared_error(y, y_pred_hist)
    mae = mean_absolute_error(y, y_pred_hist)
    r2 = r2_score(y, y_pred_hist)
    return preds, mse, mae, r2

def ml_forecast_engine():
    st.title("🚀 Enhanced ML Stock Forecast Engine")

    ticker = st.sidebar.text_input("📌 Stock Ticker", "AAPL")
    use_scaling = st.sidebar.checkbox("Use Feature Scaling", True)

    forecast_option = st.sidebar.radio("Forecast Type", ["Next N Days", "Forecast Until Date", "Date Range"])

    today = datetime.today().date()
    if forecast_option == "Next N Days":
        forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 90, 15)
        forecast_start = today + timedelta(days=1)
    elif forecast_option == "Forecast Until Date":
        forecast_end = st.sidebar.date_input("Forecast End Date", today + timedelta(days=15))
        forecast_days = (forecast_end - today).days
        forecast_start = today + timedelta(days=1)
    else:
        forecast_start = st.sidebar.date_input("Start Date", today)
        forecast_end = st.sidebar.date_input("End Date", today + timedelta(days=15))
        forecast_days = (forecast_end - forecast_start).days

    if forecast_days <= 0:
        st.error("📅 Forecast period must be at least 1 day.")
        return

    model_choices = st.sidebar.multiselect(
        "🧠 ML Models",
        ["Linear Regression", "Random Forest", "XGBoost"],
        default=["Linear Regression", "XGBoost"]
    )

    hybrid_mode = st.sidebar.checkbox("Enable Hybrid Prediction", True)
    weighted_hybrid = st.sidebar.checkbox("Use R²-Weighted Hybrid", True)

    df = fetch_data(ticker)

    future_dates = [forecast_start + timedelta(days=i) for i in range(forecast_days)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

    X_all = df[['Date_ordinal']]
    scaler = StandardScaler().fit(X_all) if use_scaling else None
    X_scaled = scaler.transform(X_all) if use_scaling else X_all.values
    future_scaled = scaler.transform(future_ordinals) if use_scaling else future_ordinals

    y = df['Close'].values

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    }

    forecast_df = pd.DataFrame({"Date": future_dates})
    performance_records, preds_list, r2_scores = [], [], []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical", line=dict(color='blue')))

    for name in model_choices:
        model = models[name]
        preds, mse, mae, r2 = forecast_prices(X_scaled, y, model, future_scaled)
        forecast_df[name] = preds
        performance_records.append({"Model": name, "MSE": mse, "MAE": mae, "R²": r2})
        preds_list.append(np.array(preds).flatten())
        r2_scores.append(max(r2, 0))
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name=name))

    if hybrid_mode and len(preds_list) > 1:
        preds_array = np.vstack(preds_list)
        weights = np.array(r2_scores) if weighted_hybrid else None
        weights = weights / weights.sum() if weights is not None else None
        hybrid_preds = np.average(preds_array, axis=0, weights=weights)
        forecast_df["Hybrid"] = hybrid_preds
        fig.add_trace(go.Scatter(x=future_dates, y=hybrid_preds, name="Hybrid Prediction", line=dict(dash='dash', width=3, color='black')))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Performance Metrics")
    perf_df = pd.DataFrame(performance_records).round(4).sort_values("R²", ascending=False)
    st.dataframe(perf_df, use_container_width=True)

    st.subheader("📋 Forecast Data")
    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")