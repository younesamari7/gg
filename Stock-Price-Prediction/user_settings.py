import streamlit as st

def user_settings():
    st.title("User Settings")
    st.subheader("Preferences")

    # Ensure base keys exist; set defaults if they don't.
    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"  # Default ticker
    if "days" not in st.session_state:
        st.session_state.days = 30  # Default forecast period

    # Initialize preferences with current settings if not already present.
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    if "default_ticker" not in st.session_state:
        st.session_state.default_ticker = st.session_state.ticker
    if "default_days" not in st.session_state:
        st.session_state.default_days = st.session_state.days

    # User input widgets with current settings as defaults.
    theme = st.radio(
        "Select Theme", 
        options=["Light", "Dark"], 
        index=0 if st.session_state.theme == "Light" else 1,
        help="Choose Light or Dark theme"
    )
    default_ticker = st.text_input(
        "Default Ticker", 
        value=st.session_state.default_ticker,
        help="Enter the default stock ticker (e.g., AAPL)"
    )
    default_days = st.number_input(
        "Default Forecast Period (days)", 
        min_value=5, 
        max_value=365, 
        value=st.session_state.default_days,
        help="Select the forecast period between 5 and 365 days"
    )

    # Save preferences back into session state.
    st.session_state.theme = theme
    st.session_state.default_ticker = default_ticker
    st.session_state.default_days = default_days
    st.session_state.ticker = default_ticker
    st.session_state.days = default_days

    st.success("Preferences saved successfully.")

    # Optional: A button to reset to original default values.
    if st.button("Reset to Defaults"):
        st.session_state.theme = "Light"
        st.session_state.default_ticker = "AAPL"
        st.session_state.default_days = 30
        st.session_state.ticker = "AAPL"
        st.session_state.days = 30
        st.experimental_rerun()  # Rerun the app to reflect changes

if __name__ == "__main__":
    user_settings()

