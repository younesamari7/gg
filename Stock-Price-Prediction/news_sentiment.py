import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

def fetch_news_with_images(query: str, api_key: str, page_size: int = 10):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "language": "en"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        st.error(f"Error fetching news from the API: {e}")
        return []

def get_sentiment_label(polarity):
    if polarity > 0.1:
        return "🟢 Positive"
    elif polarity < -0.1:
        return "🔴 Negative"
    else:
        return "🟡 Neutral"

def news_sentiment():
    # 🕒 Auto-refresh every 5 minutes (300000 ms)
    st_autorefresh(interval=300000, key="news_refresh")

    st.title("🧠 News & Sentiment Analysis")

    ticker = st.text_input("Enter Stock or Crypto Symbol", st.session_state.get("ticker", "AAPL")).upper()
    st.session_state.ticker = ticker

    api_key = "6f15cf7de3414430b24b88e64828f3ba"
    st.subheader(f"📰 Latest News for {ticker}")

    news_items = fetch_news_with_images(ticker, api_key, page_size=10)
    sentiments = []

    if news_items:
        for article in news_items:
            title = article.get("title") or ""
            description = article.get("description") or "No description provided."
            url = article.get("url")
            image_url = article.get("urlToImage")
            source = article.get("source", {}).get("name", "Unknown Source")
            date = article.get("publishedAt", "")[:10]

            if not title.strip():
                continue

            polarity = TextBlob(title).sentiment.polarity
            label = get_sentiment_label(polarity)
            sentiments.append(polarity)

            st.markdown(f"### {title}")
            st.caption(f"{source} | Published: {date}")
            if image_url:
                st.image(image_url, width=300)
            st.write(description)
            st.markdown(f"[🔗 Read more]({url})")
            st.markdown(f"**Sentiment:** {label} _(Polarity: {polarity:.2f})_")
            st.markdown("---")
    else:
        st.warning("No news articles available.")

    # 📊 Overall Sentiment
    st.subheader("📊 Overall Sentiment of Headlines")

    if sentiments:
        avg_polarity = sum(sentiments) / len(sentiments)
        overall_label = get_sentiment_label(avg_polarity)

        st.metric("Average Polarity", f"{avg_polarity:.2f}", help="Mean sentiment polarity across top 10 articles")
        st.markdown(f"**Overall Sentiment:** {overall_label}")

        st.bar_chart(pd.Series(sentiments, name="Headline Sentiment"))
        st.caption("Note: Polarity ranges from -1 to 1. Positive = 🟢, Neutral = 🟡, Negative = 🔴")
    else:
        st.info("No valid headlines to compute sentiment.")

if __name__ == "__main__":
    news_sentiment()
