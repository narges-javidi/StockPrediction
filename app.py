import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from textblob import TextBlob
# فرض می کنیم این ماژول برای بورس تهران تعریف شده است.
# اگر این ماژول در Render موجود نیست، بخش مربوط به آن را کامنت کنید.
from data.tse_data import get_tse_data 

# --- توابع کمکی (Auxiliary Functions) ---

def fetch_stock_data(symbol):
    API_KEY = 'NA3UHC1XJU4OQKKO'  # Replace with your API key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={API_KEY}'
    
    response = requests.get(url)
    
    try:
        data = response.json()
    except Exception as e:
        st.error(f"Error parsing JSON for {symbol}")
        print("Raw response:", response.text)
        return None

    if 'Monthly Time Series' not in data:
        st.error(f"Alpha Vantage error for {symbol}: {data.get('Note') or data.get('Error Message') or data}")
        return None

    time_series = data['Monthly Time Series']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df[['4. close']]
    df.columns = ['Close']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def fetch_news(stock_ticker):
    api_key = "91fc8cf73730404fb5b9c38c67038870"  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&sortBy=publishedAt&apiKey={api_key}&language=en"
    
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get('articles', [])
    else:
        st.error("Failed to fetch news. Check your API key or network.")
        return []

def preprocess_data(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = calculate_volatility(df)
    df = df.dropna()
    return df


def train_model(df):
    features = ['SMA_50', 'SMA_200', 'Price_Change']
    X = df[features]
    y = df['Close']
    

    if X.empty or y.empty:
        st.error("No data available for training the model. Please check the stock data.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model

def predict_stock_price(model, df):
    features = ['SMA_50', 'SMA_200', 'Price_Change']
    X = df[features]
    predictions = model.predict(X)
    return predictions

def analyze_sentiment(article_title):
    analysis = TextBlob(article_title)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive", "green"
    elif polarity == 0:
        return "Neutral", "gray"
    else:
        return "Negative", "red"
    
def calculate_moving_average(data, short_window=20, long_window=50):
    if 'Close' not in data.columns:
        raise ValueError("The input data must contain a 'Close' column.")
    
    moving_averages = pd.DataFrame(index=data.index)
    moving_averages['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    moving_averages['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    return moving_averages

def calculate_volatility(data, window=30):
    if 'Close' not in data.columns:
        raise ValueError("The input data must contain a 'Close' column.")
    
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = log_returns.rolling(window=window).std()
    
    return volatility

def analyze_sentiment_and_recommendation(ticker):
    news_articles = fetch_news(ticker)
    
    sentiments = []
    for article in news_articles[:5]:
        sentiment, color = analyze_sentiment(article['title'])
        sentiments.append((sentiment, color))

    return sentiments

def make_recommendation(predicted_price, actual_price, model, processed_data):
    predicted_change = predicted_price - actual_price

    X = processed_data.drop(columns='Close')
    y = processed_data['Close']
    r2_score = model.score(X, y)

    recommendation = "Hold"
    color = "orange"

    if r2_score > 0.8:
        if predicted_change > 0:
            recommendation = "Buy"
            color = "green"
        elif predicted_change < 0:
            recommendation = "Sell"
            color = "red"
        else:
            recommendation = "Hold"
            color = "orange"
    else:
        recommendation = "Hold"
        color = "orange"
    
    return recommendation, color, r2_score


#-------- Initialization Block for Streamlit Session State --------
# این بخش جدید است و اطمینان می دهد که state قبل از استفاده مقداردهی می شود.

# 1. Navigation State
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# 2. Widget States (Keys used in the UI)
# اینها کلیدهایی هستند که در بخش های مختلف برنامه استفاده می شوند.
if 'ticker_select' not in st.session_state:
    st.session_state.ticker_select = 'AAPL'
if 'compare_checkbox' not in st.session_state:
    st.session_state.compare_checkbox = False
if 'tickers_compare' not in st.session_state:
    st.session_state.tickers_compare = ['AAPL', 'GOOGL']
if 'news_ticker_select' not in st.session_state:
    st.session_state.news_ticker_select = 'AAPL'

# 3. Data/Model Caching State (Optional but good practice)
# اگرچه در کد شما از کش مبتنی بر فایل استفاده شده، اما تعریف اینجا به جلوگیری از ارور کمک می کند.
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

# -------- End of Initialization Block --------

# --- تنظیمات صفحه و استایل ها ---
st.set_page_config(page_title="Stock Predictor", page_icon="📈 ", layout="wide")

st.markdown(
    """
    <style>
        .main {
            max-width: 90%; 
            margin: 0 auto;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# --- Style Definitions (Copied from your original code) ---
st.markdown("""
    <style>
        /* Global Font and Body Styling */
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: black;
        }
        
        /* Title Styling */
        .title {
            font-size: 48px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: blue;
            text-align: center;
            font-weight: bold;
            margin-top: 50px;
            letter-spacing: 1px;
        }
        
        /* Subtitle Styling */
        .subtitle {
            font-size: 22px;
            font-family: 'Arial', sans-serif;
            color: #7f8c8d;
            text-align: center;
            font-weight: normal;
            margin-top: -10px;
        }
        
    </style>
    """, unsafe_allow_html=True)
# --- End Style Definitions ---

# Buttons for navigation (Using session_state values)
if st.sidebar.button("🏠 Home Page"):
    st.session_state.current_page = 'home'
if st.sidebar.button("📊 Stock Info"):
    st.session_state.current_page = 'stock_info'
if st.sidebar.button("📰 Stock News"):
    st.session_state.current_page = 'stock_news'

# Display content based on current page
if st.session_state.current_page == 'home':

    # Sample data for candlestick chart
    data = {
        'Date': pd.date_range(start='2024-12-01', periods=6, freq='D'),
        'Open': [150, 155, 160, 162, 165, 167],
        'High': [155, 160, 165, 167, 170, 172],
        'Low': [148, 152, 157, 160, 163, 165],
        'Close': [153, 158, 162, 164, 168, 170]
    }
    
    df = pd.DataFrame(data)
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        title="Stock Price Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )
    
    st.markdown("<h1 class='title'>📈 Advanced Stock Price Prediction App 📉</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your Professional Tool for Stock Analysis and Market Insights 🚀</p>", unsafe_allow_html=True)

    st.plotly_chart(fig)
       
        

elif st.session_state.current_page == 'stock_info':
    # Stock info page content
    st.markdown(
        "<p style='text-align: center; font-size: 22px; color: #264653;'>Welcome to the Stock Price Prediction App!</p>", 
        unsafe_allow_html=True
    )
    
    # Dropdown for stock symbol selection (Uses session_state key)
    ticker = st.selectbox(
        "🎯 **Select a Stock Symbol**", 
        stock_symbols,
        help="Choose the stock you want to analyze and predict.",
        key="ticker_select"
    )

    # Checkbox to compare multiple stocks (Uses session_state key)
    comparison = st.checkbox('Compare multiple stocks', key="compare_checkbox")
    if comparison:
        tickers_to_compare = st.multiselect(
            "📊 **Select Stocks to Compare**", 
            stock_symbols, 
            default=['AAPL', 'GOOGL'],
            help="Compare predictions for multiple stocks.",
            key="tickers_compare"
        )
    else:
         tickers_to_compare = [ticker] 
    
    predictions_dict = {}

    CACHE_DIR = 'cached_stock_data'
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    def save_to_cache(ticker, data):
        data.to_csv(os.path.join(CACHE_DIR, f"{ticker}.csv"))

    def load_from_cache(ticker):
        file_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        return None

    def create_stock_plot(data, predictions, ticker):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=predictions, mode='lines', name='Predicted Price', line=dict(color='#e76f51', dash='dash')))
        fig.update_layout(
            title=f"📈 {ticker} Stock Price Prediction",
            xaxis_title="📅 Date",
            yaxis_title="💲 Stock Price (USD)",
            template="plotly_dark",
            title_font=dict(size=22, color='#e9c46a'),
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        return fig

    # Fetch data and predict for selected stocks
    for ticker in tickers_to_compare:
        # NOTE: In a real deployment, file I/O (like caching) can be tricky in serverless environments. 
        # If this caching causes issues on Render, remove the load/save functions and always fetch the data.
        cached_data = load_from_cache(ticker)

        if cached_data is not None:
            data = cached_data
        else:
            data = fetch_stock_data(ticker)
            if data is not None and not data.empty:
                save_to_cache(ticker, data)
            else:
                st.error(f"❌ Failed to fetch data for {ticker}. Please check the stock ticker.")
                continue

        processed_data = preprocess_data(data)
        if processed_data.empty:
            st.error(f"No valid data available for {ticker} after preprocessing.")
            continue
        model = train_model(processed_data)
        if model is None:
           continue
        predictions = predict_stock_price(model, processed_data)

        predictions_dict[ticker] = predictions[-1]

        st.plotly_chart(create_stock_plot(processed_data, predictions, ticker))
        
    if predictions_dict:
        highest_stock = max(predictions_dict, key=predictions_dict.get)
        lowest_stock = min(predictions_dict, key=predictions_dict.get)

        st.markdown("### 📊 **Stock Performance Summary:**")
        st.write(
            f"🌟 **Stock with the highest predicted price:** `{highest_stock}` - **${predictions_dict[highest_stock]:.2f}**"
        )
        st.write(
            f"📉 **Stock with the lowest predicted price:** `{lowest_stock}` - **${predictions_dict[lowest_stock]:.2f}**"
        )

        st.markdown("### 💡 **Recommendations:**")
        for ticker in tickers_to_compare:
            # Check if data was successfully processed for this ticker before accessing its index
            try:
                # Re-fetch data to get the last actual closing price for comparison
                current_data = fetch_stock_data(ticker) 
                if current_data is None or current_data.empty: continue
                last_close_price = current_data['Close'].iloc[-1]

                predicted_price = predictions_dict.get(ticker)
                if predicted_price is None: continue

                predicted_change = predicted_price - last_close_price

                if predicted_change > 0:
                    recommendation = "Buy"
                    color = "#2b9c5a"
                elif predicted_change == 0:
                    recommendation = "Hold"
                    color = "#f4a300"
                else:
                    recommendation = "Sell"
                    color = "#e63946"

                st.markdown(f"""
                    <div style="
                        display: inline-block;
                        padding: 10px 20px;
                        margin: 5px;
                        background-color: {color};
                        color: white;
                        font-weight: bold;
                        border-radius: 8px;
                        text-align: center;
                        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
                    ">
                        {ticker}: {recommendation} (${predicted_change:.2f})
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                # Catch errors if data loading fails mid-loop
                st.warning(f"Could not generate recommendation for {ticker} due to data loading issue.")


elif st.session_state.current_page == 'stock_news':
    # Stock news page content
    news_ticker = st.selectbox(
        "🎯 **Select a Stock for News**", 
        stock_symbols, 
        help="Choose the stock for which you want to view news and sentiment analysis.",
        key="news_ticker_select"
    )

    st.markdown(f"### 📰 **Latest News and Sentiment Analysis for {news_ticker}**")
    news_articles = fetch_news(news_ticker)

    if news_articles:
        for article in news_articles[:5]:
            title = article['title']
            url = article['url']
            source = article['source']['name']

            sentiment, color = analyze_sentiment(title)

            st.markdown(
                f"""
                <div style="
                    display: flex; 
                    justify-content: space-between; 
                    align-items: center; 
                    border: 2px solid {color}; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 10px 0; 
                    background-color: rgba(255, 255, 255, 0.05);
                ">
                    <div style="flex: 1;">
                        <b><a href="{url}" style="text-decoration: none; color: #264653;" target="_blank">{title}</a></b>
                        <p style="color: #264653; margin: 5px 0;">Source: {source}</p>
                    </div>
                    <div>
                        <button style="
                            background-color: {color};
                            border: none;
                            border-radius: 8px;
                            color: white;
                            padding: 8px 12px;
                            cursor: pointer;
                        ">
                            {sentiment}
                        </button>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No news articles found. 📭")

st.markdown("---")
st.subheader("📊 تست اتصال به بورس تهران")

if st.button("دریافت دیتای فولاد"):
    # این بخش به دلیل ماهیت محیط Render ممکن است نیاز به کتابخانه های خاص داشته باشد.
    try:
        df = get_tse_data("فولاد")
        st.write(df.head())
    except NameError:
        st.error("ماژول 'data.tse_data' یا تابع 'get_tse_data' تعریف نشده یا در محیط Render قابل دسترسی نیست.")
    except Exception as e:
        st.error(f"خطا در دریافت داده از بورس تهران: {e}")
