import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
import plotly.graph_objects as go
import plotly.express as px

def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy between true and predicted values."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    if min_len < 2:
        return np.nan
    # Use only up to the shortest length
    true_dir = np.sign(np.diff(y_true[:min_len]))
    pred_dir = np.sign(np.diff(y_pred[:min_len]))
    correct = np.sum(true_dir == pred_dir)
    return correct / len(true_dir) if len(true_dir) > 0 else np.nan

def fetch_data(ticker, period="5y"):
    df = yf.download(ticker, period=period)[['Close']].dropna().reset_index()
    df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    return df

def forecast_prices(X, y, model, future_scaled):
    y = np.asarray(y).flatten()   # Fix: ensure y is always 1D
    model.fit(X, y)
    preds = model.predict(future_scaled)
    y_pred_hist = model.predict(X)
    mse = mean_squared_error(y, y_pred_hist)
    mae = mean_absolute_error(y, y_pred_hist)
    r2 = r2_score(y, y_pred_hist)
    dir_acc = direction_accuracy(y, y_pred_hist)
    return preds, mse, mae, r2, dir_acc

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

    # Model choices for sidebar (add new models here)
    model_options = [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "SVR",
        "KNN",
        "Ridge",
        "Lasso"
    ]
    if LGBMRegressor:
        model_options.append("LightGBM")
    if CatBoostRegressor:
        model_options.append("CatBoost")

    model_choices = st.sidebar.multiselect(
        "🧠 ML Models",
        model_options,
        default=["Linear Regression", "XGBoost"]
    )

    # Option to enable grid search for Random Forest
    rf_grid_search = st.sidebar.checkbox("Grid Search for Random Forest", False)

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

    # Initialize all models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Ridge": Ridge(),
        "Lasso": Lasso()
    }
    if LGBMRegressor:
        models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=42)
    if CatBoostRegressor:
        models["CatBoost"] = CatBoostRegressor(n_estimators=200, random_state=42, verbose=0)

    # Parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 8, 16, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['auto', 'sqrt', 0.5]
    }

    forecast_df = pd.DataFrame({"Date": future_dates})
    performance_records, preds_list, r2_scores = [], [], []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical", line=dict(color='blue')))

    for name in model_choices:
        # Grid search for Random Forest if enabled
        if name == "Random Forest" and rf_grid_search:
            grid = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
            grid.fit(X_scaled, np.asarray(y).flatten())
            model = grid.best_estimator_
            st.write("Best Random Forest parameters:", grid.best_params_)
        else:
            model = models[name]

        preds, mse, mae, r2, dir_acc = forecast_prices(X_scaled, y, model, future_scaled)
        forecast_df[name] = preds
        performance_records.append({"Model": name, "MSE": mse, "MAE": mae, "R²": r2, "Direction_Accuracy": dir_acc})
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
