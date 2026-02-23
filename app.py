import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from textblob import TextBlob
from data.tse_data import get_tse_data

# Fetch real-time stock data from Alpha Vantage (monthly data in this case)
# def fetch_stock_data(symbol):
#     API_KEY = 'NA3UHC1XJU4OQKKO'  # Replace with your Alpha Vantage API key
#     url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={API_KEY}'
    
#     response = requests.get(url)
#     data = response.json()
    
#     # Check if the data contains 'Monthly Time Series'
#     if 'Monthly Time Series' in data:
#         time_series = data['Monthly Time Series']
#         # Convert the time series data into a pandas DataFrame
#         df = pd.DataFrame.from_dict(time_series, orient='index')
#         df = df[['4. close']]  # We only need the 'close' price
#         df.columns = ['Close']
        
#         # Convert 'Close' column to numeric (float)
#         df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Convert to float, errors='coerce' will turn invalid values to NaN
        
#         df.index = pd.to_datetime(df.index)  # Convert the index to datetime
#         df = df.sort_index()  # Sort by date
        
#         return df
#     else:
#         print("Error fetching data. Please check the symbol or try again.")
#         return None

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

    # Log the raw response if data is missing
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
# Preprocess stock data (e.g., use moving averages, lagged features)
def preprocess_data(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # Simple Moving Average (50 days)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # Simple Moving Average (200 days)
    df['Price_Change'] = df['Close'].pct_change()  # Daily price change percentage
    df['Volatility'] = calculate_volatility(df)  # Add Volatility as a feature
    df = df.dropna()  # Drop rows with NaN values
    return df


# Train a RandomForestRegressor model
def train_model(df):
    features = ['SMA_50', 'SMA_200', 'Price_Change']
    X = df[features]
    y = df['Close']
    

    # Check if the DataFrame is empty
    if X.empty or y.empty:
        st.error("No data available for training the model. Please check the stock data.")
        return None  # Return None if there's no data
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model

# Predict stock price using the trained model
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
def analyze_sentiment_and_recommendation(ticker):
    news_articles = fetch_news(ticker)
    
    sentiments = []
    for article in news_articles[:5]:  # Top 5 news
        sentiment, color = analyze_sentiment(article['content'])  # Use 'content' instead of 'title'
        sentiments.append((sentiment, color))

    return sentiments

def calculate_moving_average(data, short_window=20, long_window=50):
    """
    Calculate short-term and long-term moving averages for the stock data.

    Parameters:
    - data: pandas DataFrame with stock data (must include 'Close' column).
    - short_window: The window size for the short-term moving average.
    - long_window: The window size for the long-term moving average.

    Returns:
    - pandas DataFrame containing moving averages.
    """
    if 'Close' not in data.columns:
        raise ValueError("The input data must contain a 'Close' column.")
    
    moving_averages = pd.DataFrame(index=data.index)
    moving_averages['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    moving_averages['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    return moving_averages

def calculate_volatility(data, window=30):
    """
    Calculate the rolling volatility for the stock data.

    Parameters:
    - data: pandas DataFrame with stock data (must include 'Close' column).
    - window: The rolling window size for calculating volatility.

    Returns:
    - pandas Series containing the volatility values.
    """
    if 'Close' not in data.columns:
        raise ValueError("The input data must contain a 'Close' column.")
    
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = log_returns.rolling(window=window).std()
    
    return volatility

def analyze_sentiment_and_recommendation(ticker):
    # Fetch the latest news for the stock
    news_articles = fetch_news(ticker)
    
    sentiments = []
    for article in news_articles[:5]:  # Top 5 news
        sentiment, color = analyze_sentiment(article['title'])
        sentiments.append((sentiment, color))

    return sentiments

# Recommendation with confidence based on model performance
def make_recommendation(predicted_price, actual_price, model, processed_data):
    predicted_change = predicted_price - actual_price

    # Calculate model performance (R² score)
    X = processed_data.drop(columns='Close')
    y = processed_data['Close']
    r2_score = model.score(X, y)  # R² score of the model

    recommendation = "Hold"  # Default
    color = "orange"  # Default color

    if r2_score > 0.8:  # High confidence
        if predicted_change > 0:
            recommendation = "Buy"
            color = "green"
        elif predicted_change < 0:
            recommendation = "Sell"
            color = "red"
        else:
            recommendation = "Hold"
            color = "orange"
    else:  # Low confidence
        recommendation = "Hold"
        color = "orange"
    
    return recommendation, color, r2_score



#--------app.py---------

st.set_page_config(page_title="Stock Predictor", page_icon="📈 ", layout="wide")

# Predefined list of stock symbols
stock_symbols = [
    'AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'FB', 'NVDA', 'SPY', 'NFLX', 'AMD',
    'IBM', 'BABA', 'NFLX', 'BA', 'GS', 'MS', 'GE', 'INTC', 'DIS', 'TSM',
    'PYPL', 'SNAP', 'UBER', 'V', 'WMT', 'SPY', 'NVDA', 'SQ', 'LUV', 'CVX',
    'XOM', 'MCD', 'PG', 'JPM', 'INTC', 'AMD', 'UNH', 'COP', 'PEP', 'KO', 'T'
]

# Streamlit app title
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

# Default view is home page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Buttons for navigation
if st.sidebar.button("🏠 Home Page"):
    st.session_state.current_page = 'home'
if st.sidebar.button("📊 Stock Info"):
    st.session_state.current_page = 'stock_info'
if st.sidebar.button("📰 Stock News"):
    st.session_state.current_page = 'stock_news'

# Display content based on current page
if st.session_state.current_page == 'home':

    # Title and Subtitle with updated professional styles
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

    # Main container with background and styling

    # Title and Subtitle
    # st.markdown("<h1 class='title'>📈 Advanced Stock Price Prediction App 📉</h1>", unsafe_allow_html=True)
    # st.markdown("<p class='subtitle'>Your Professional Tool for Stock Analysis and Market Insights 🚀</p>", unsafe_allow_html=True)
    # Sample data for candlestick chart (you can replace this with real data or use more points)
    data = {
        'Date': pd.date_range(start='2024-12-01', periods=6, freq='D'),
        'Open': [150, 155, 160, 162, 165, 167],
        'High': [155, 160, 165, 167, 170, 172],
        'Low': [148, 152, 157, 160, 163, 165],
        'Close': [153, 158, 162, 164, 168, 170]
    }
    
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)
    
    # Create a candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    # Update layout for better presentation
    fig.update_layout(
        title="Stock Price Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",  # You can change the theme
        xaxis_rangeslider_visible=False,  # Remove range slider
    )
    
    # Display the chart in Streamlit
    # st.title("Welcome to Stock Market Dashboard")
    st.markdown("<h1 class='title'>📈 Advanced Stock Price Prediction App 📉</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your Professional Tool for Stock Analysis and Market Insights 🚀</p>", unsafe_allow_html=True)

    st.plotly_chart(fig)
       
        

elif st.session_state.current_page == 'stock_info':
    # Stock info page content
    st.markdown(
        "<p style='text-align: center; font-size: 22px; color: #264653;'>Welcome to the Stock Price Prediction App!</p>", 
        unsafe_allow_html=True
    )
    
    # Dropdown for stock symbol selection
    ticker = st.selectbox(
        "🎯 **Select a Stock Symbol**", 
        stock_symbols,
        help="Choose the stock you want to analyze and predict.",
        key="ticker_select"
    )

    # Checkbox to compare multiple stocks
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
    
    # Initialize a dictionary to store the predicted prices
    predictions_dict = {}

    # Directory to store cached data
    CACHE_DIR = 'cached_stock_data'
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Function to cache stock data locally
    def save_to_cache(ticker, data):
        data.to_csv(os.path.join(CACHE_DIR, f"{ticker}.csv"))

    # Function to load cached stock data
    def load_from_cache(ticker):
        file_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        return None

    # Function to create stock prediction plot
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
        # Try to load cached data
        cached_data = load_from_cache(ticker)

        if cached_data is not None:
            data = cached_data
        else:
            # If data is not cached, fetch from the API
            data = fetch_stock_data(ticker)
            if data is not None and not data.empty:
                save_to_cache(ticker, data)
            else:
                st.error(f"❌ Failed to fetch data for {ticker}. Please check the stock ticker.")
                continue

        # Preprocess data and make predictions
        processed_data = preprocess_data(data)
        if processed_data.empty:
            st.error(f"No valid data available for {ticker} after preprocessing.")
            continue
        model = train_model(processed_data)
        if model is None:
           continue
        predictions = predict_stock_price(model, processed_data)

        # Store predictions
        predictions_dict[ticker] = predictions[-1]  # Store the last predicted price

        # Display prediction plot
        st.plotly_chart(create_stock_plot(processed_data, predictions, ticker))
    # Display predictions for all selected stocks
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

        # Recommendations with custom styling
        st.markdown("### 💡 **Recommendations:**")
        for ticker in tickers_to_compare:
            # Get the last actual closing price
            last_close_price = data['Close'].iloc[-1]

            # Calculate the price change
            predicted_change = predictions_dict[ticker] - last_close_price

            # Determine recommendation
            if predicted_change > 0:
                recommendation = "Buy"
                color = "#2b9c5a"  # Green for Buy
            elif predicted_change == 0:
                recommendation = "Hold"
                color = "#f4a300"  # Orange for Hold
            else:
                recommendation = "Sell"
                color = "#e63946"  # Red for Sell

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
        for article in news_articles[:5]:  # Show top 5 news articles
            title = article['title']
            url = article['url']
            source = article['source']['name']

            # Analyze sentiment
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
    df = get_tse_data("فولاد")
    st.write(df.head())


