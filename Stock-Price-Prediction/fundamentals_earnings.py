import streamlit as st
import yfinance as yf
import pandas as pd

def fundamentals_earnings():
    st.title("📚 Fundamentals, Earnings & Financials Dashboard")

    # --- Symbol Input ---
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", st.session_state.get("ticker", "AAPL")).upper()
    st.session_state.ticker = ticker

    try:
        asset = yf.Ticker(ticker)
        info = asset.info

        # === Valuation Metrics ===
        st.subheader(f"📊 Valuation Metrics for {ticker}")
        fundamental_data = {
            "Previous Close": info.get("previousClose", "N/A"),
            "Open": info.get("open", "N/A"),
            "PE Ratio (TTM)": info.get("trailingPE", "N/A"),
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "Dividend Yield": f"{round(info['dividendYield'] * 100, 2)}%" if info.get("dividendYield") else "N/A"
        }

        df_fundamentals = pd.DataFrame.from_dict(fundamental_data, orient='index', columns=["Value"])
        st.table(df_fundamentals)

        with st.expander("📘 What These Metrics Mean"):
            st.markdown("""
            - **Previous Close**: Last traded price before today's market opens.  
            - **Open**: Opening price today.  
            - **PE Ratio**: Shows how expensive a stock is relative to its earnings.  
            - **EPS**: Profit per outstanding share.  
            - **Dividend Yield**: Income as a % of current stock price.
            """)

        st.markdown("---")

        # === Earnings Info ===
        st.subheader(f"🗓️ Earnings Info for {ticker}")
        earnings_date = info.get("nextEarningsDate") or info.get("earningsDate") or "N/A"

        if earnings_date == "N/A":
            st.info("No upcoming earnings date available.")
        else:
            st.success(f"**Next Earnings Date:** {earnings_date}")

        with st.expander("📘 Why It Matters"):
            st.markdown("""
            - Companies report earnings quarterly.  
            - These reports often move the stock price.  
            - Watch for earnings surprises or strong guidance.
            """)

        st.markdown("---")

        # === Balance Sheet ===
        st.subheader("📈 Balance Sheet")
        try:
            bs = asset.balance_sheet
            if not bs.empty:
                st.dataframe(bs.transpose())
            else:
                st.info("Balance sheet data not available.")
        except:
            st.warning("Unable to fetch balance sheet.")

        # === Income Statement ===
        st.subheader("💰 Income Statement")
        try:
            income = asset.financials
            if not income.empty:
                st.dataframe(income.transpose())
            else:
                st.info("Income statement data not available.")
        except:
            st.warning("Unable to fetch income statement.")

        # === Analyst Price Target ===
        st.subheader("🎯 Analyst Price Target")
        try:
            target_low = info.get("targetLowPrice", None)
            target_mean = info.get("targetMeanPrice", None)
            target_high = info.get("targetHighPrice", None)

            if target_mean:
                col1, col2, col3 = st.columns(3)
                col1.metric("Target Low", f"${target_low:.2f}" if target_low else "N/A")
                col2.metric("Target Mean", f"${target_mean:.2f}")
                col3.metric("Target High", f"${target_high:.2f}" if target_high else "N/A")

                with st.expander("📘 What It Means"):
                    st.markdown("""
                    - These targets are based on analyst estimates.  
                    - They help investors understand how professionals view future price potential.  
                    - **Mean price** is the average of all analyst estimates.
                    """)
            else:
                st.info("No analyst target data available.")
        except:
            st.warning("Unable to fetch analyst target information.")

    except Exception as e:
        st.error(f"⚠️ Error retrieving data for '{ticker}': {e}")

# Run standalone
if __name__ == "__main__":
    fundamentals_earnings()
