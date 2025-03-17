import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
from newsapi import NewsApiClient
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob  # For sentiment analysis
import plotly.express as px  # For interactive charts

# ----------------------------
# 1. Get S&P 500 Tickers from Wikipedia
# ----------------------------
def get_sp500_tickers():
    """
    Fetch the list of S&P 500 tickers from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(response.text)
    sp500_table = tables[0]  # The first table contains the S&P 500 tickers
    tickers = sp500_table['Symbol'].tolist()

    # Replace special characters in tickers (e.g., BRK.B -> BRK-B)
    tickers = [ticker.replace(".", "-") for ticker in tickers]

    print(f"Fetched {len(tickers)} S&P 500 tickers.")
    return tickers

# Get the list of S&P 500 tickers
sp500_tickers = get_sp500_tickers()

# ----------------------------
# 2. Fetch Data from Yahoo Finance
# ----------------------------
def fetch_yahoo_data(tickers):
    """
    Fetch historical stock data from Yahoo Finance for multiple tickers.
    """
    # Define date range
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch data for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    print("Yahoo Finance data fetched successfully.")
    return data

# Fetch data for all S&P 500 companies
yahoo_df = fetch_yahoo_data(sp500_tickers)

# ----------------------------
# 3. Fetch News Sentiment Data Using NewsAPI
# ----------------------------
def fetch_news_sentiment(tickers, api_key):
    """
    Fetch news articles for the given tickers using NewsAPI.
    """
    newsapi = NewsApiClient(api_key=api_key)
    news_data = []

    for ticker in tickers:
        try:
            # Fetch top headlines for the ticker
            headlines = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=100)

            # Append news articles to the list
            for article in headlines['articles']:
                news_data.append({
                    'ticker': ticker,
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'published_at': article['publishedAt']  # Ensure this matches the API response
                })
            print(f"Fetched news articles for {ticker}.")
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")

    # Convert news data to a DataFrame
    news_df = pd.DataFrame(news_data)

    # Debugging: Print columns and a sample of the data
    print("Columns in news_df:", news_df.columns)
    print("Sample of news_df:", news_df.head())

    return news_df

# Fetch news sentiment data for S&P 500 companies
NEWSAPI_API_KEY = "8a3c5d1c894e4377ab68c1c55206bcc2"  # Replace with your NewsAPI key
news_df = fetch_news_sentiment(sp500_tickers, NEWSAPI_API_KEY)

# ----------------------------
# 4. Fetch Earnings Data from Alpha Vantage
# ----------------------------
def fetch_alpha_vantage_data(tickers, api_key):
    """
    Fetch earnings data from Alpha Vantage for multiple tickers.
    """
    earnings_data = []

    for ticker in tickers:
        try:
            url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}"
            response = requests.get(url)
            data = response.json()

            # Debugging: Print the API response
            print(f"API response for {ticker}:", data)

            # Append earnings data to the list
            for earnings in data.get("quarterlyEarnings", []):
                earnings_data.append({
                    'ticker': ticker,
                    'fiscalDateEnding': earnings['fiscalDateEnding'],
                    'reportedEPS': earnings['reportedEPS'],
                    'estimatedEPS': earnings['estimatedEPS'],
                    'surprise': earnings['surprise'],
                    'surprisePercentage': earnings['surprisePercentage']
                })
            print(f"Fetched earnings data for {ticker}.")
        except Exception as e:
            print(f"Error fetching earnings data for {ticker}: {e}")

    # Convert earnings data to a DataFrame
    earnings_df = pd.DataFrame(earnings_data)

    # Debugging: Print columns and a sample of the data
    print("Columns in earnings_df:", earnings_df.columns)
    print("Sample of earnings_df:", earnings_df.head())

    return earnings_df

# Fetch earnings data for S&P 500 companies
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"  # Replace with your Alpha Vantage key
earnings_df = fetch_alpha_vantage_data(sp500_tickers, ALPHA_VANTAGE_API_KEY)

# ----------------------------
# 5. Fetch Macroeconomic Data from FRED
# ----------------------------
def fetch_fred_data(series_id, start_date, end_date):
    """
    Fetch macroeconomic data from FRED.
    """
    try:
        data = web.DataReader(series_id, "fred", start_date, end_date)
        print(f"FRED data for {series_id} fetched successfully.")
        return data
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return None

# Define date range for FRED data
start_date = datetime(2020, 1, 1)
end_date = datetime.now()

# Fetch GDP data from FRED
fred_df = fetch_fred_data("GDP", start_date, end_date)

# ----------------------------
# 6. Preprocess Data
# ----------------------------
def preprocess_data(stock_df, news_df, earnings_df, fred_df):
    """
    Preprocess the data from all APIs.
    """
    # Preprocess stock data
    stock_df = stock_df.ffill().dropna()  # Fill missing values and drop rows with missing values

    # Preprocess news data
    if 'published_at' in news_df.columns:
        news_df['published_at'] = pd.to_datetime(news_df['published_at'])  # Convert to datetime
        news_df = news_df.dropna(subset=['title', 'description'])  # Drop rows with missing titles or descriptions
    else:
        print("Warning: 'published_at' column not found in news_df. Skipping news data preprocessing.")

    # Preprocess earnings data
    if 'fiscalDateEnding' in earnings_df.columns:
        earnings_df['fiscalDateEnding'] = pd.to_datetime(earnings_df['fiscalDateEnding'])  # Convert to datetime
        earnings_df = earnings_df.dropna()  # Drop rows with missing values
    else:
        print("Warning: 'fiscalDateEnding' column not found in earnings_df. Skipping earnings data preprocessing.")

    # Preprocess FRED data
    fred_df = fred_df.ffill().dropna()  # Fill missing values and drop rows with missing values

    print("Data preprocessing complete.")
    return stock_df, news_df, earnings_df, fred_df

# Preprocess the data
preprocessed_stock_df, preprocessed_news_df, preprocessed_earnings_df, preprocessed_fred_df = preprocess_data(
    yahoo_df, news_df, earnings_df, fred_df
)

# ----------------------------
# Summary
# ----------------------------
print("\nAll data fetched and preprocessed.")

#-----------------------------
#Data Visualization
#-----------------------------


# ----------------------------
# 1. Visualize Stock Price Trends for Selected Companies
# ----------------------------
def visualize_stock_trends(df, tickers):
    """
    Visualize stock price trends for selected companies.
    """
    if df is not None and len(df.columns) > 0:
        plt.figure(figsize=(14, 8))
        for ticker in tickers:
            if ticker in df.columns.get_level_values(0):
                plt.plot(df[ticker]['Close'], label=ticker)
        plt.title("Stock Price Trends for Selected S&P 500 Companies")
        plt.xlabel("Date")
        plt.ylabel("Close Price ($)")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Warning: Stock data is missing or empty. Skipping stock price trends visualization.")

# Example: Visualize trends for a few companies
selected_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
visualize_stock_trends(preprocessed_stock_df, selected_tickers)

# ----------------------------
# 2. Additional Charts for Stock Data
# ----------------------------
def visualize_stock_analysis(df, tickers):
    """
    Visualize additional stock analysis charts.
    """
    if df is not None and len(df.columns) > 0:
        for ticker in tickers:
            if ticker in df.columns.get_level_values(0):
                # Daily returns distribution
                daily_returns = df[ticker]['Close'].pct_change().dropna()
                plt.figure(figsize=(10, 6))
                plt.hist(daily_returns, bins=50, color='blue', alpha=0.7)
                plt.title(f"Daily Returns Distribution for {ticker}")
                plt.xlabel("Daily Return")
                plt.ylabel("Frequency")
                plt.grid()
                plt.show()

                # Moving averages
                moving_avg_50 = df[ticker]['Close'].rolling(window=50).mean()
                moving_avg_200 = df[ticker]['Close'].rolling(window=200).mean()

                plt.figure(figsize=(12, 6))
                plt.plot(df[ticker]['Close'], label='Close Price')
                plt.plot(moving_avg_50, label='50-Day Moving Average', linestyle='--')
                plt.plot(moving_avg_200, label='200-Day Moving Average', linestyle='--')
                plt.title(f"Moving Averages for {ticker}")
                plt.xlabel("Date")
                plt.ylabel("Price ($)")
                plt.legend()
                plt.grid()
                plt.show()
    else:
        print("Warning: Stock data is missing or empty. Skipping stock analysis visualization.")

# Visualize additional stock analysis
visualize_stock_analysis(preprocessed_stock_df, selected_tickers)

# ----------------------------
# 3. Perform Sentiment Analysis on News Data
# ----------------------------
def analyze_sentiment(news_df):
    """
    Perform sentiment analysis on news articles.
    """
    if news_df is not None and 'title' in news_df.columns:
        # Calculate sentiment polarity for each article
        news_df['sentiment'] = news_df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        return news_df
    else:
        print("Warning: News data is missing or does not contain 'title' column. Skipping sentiment analysis.")
        return None

# Analyze sentiment for news data
news_with_sentiment = analyze_sentiment(preprocessed_news_df)

# Visualize sentiment distribution
if news_with_sentiment is not None and 'sentiment' in news_with_sentiment.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(news_with_sentiment['sentiment'], bins=50, color='blue', alpha=0.7)
    plt.title("Sentiment Distribution of News Articles")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
else:
    print("Warning: Sentiment data is missing or empty. Skipping sentiment distribution visualization.")

# ----------------------------
# 4. Additional Charts for News Data
# ----------------------------
def visualize_news_analysis(news_df):
    """
    Visualize additional news analysis charts.
    """
    if news_df is not None and 'published_at' in news_df.columns:
        # Sentiment over time
        news_df['date'] = pd.to_datetime(news_df['published_at']).dt.date
        daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sentiment['date'], daily_sentiment['sentiment'], marker='o', color='green')
        plt.title("Average Sentiment Over Time")
        plt.xlabel("Date")
        plt.ylabel("Sentiment Polarity")
        plt.grid()
        plt.xticks(rotation=45)
        plt.show()

        # Top companies by news volume
        top_companies = news_df['ticker'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        top_companies.plot(kind='bar', color='purple')
        plt.title("Top Companies by News Volume")
        plt.xlabel("Ticker")
        plt.ylabel("Number of Articles")
        plt.grid()
        plt.show()
    else:
        print("Warning: News data is missing or does not contain 'published_at' column. Skipping news analysis visualization.")

# Visualize additional news analysis
visualize_news_analysis(news_with_sentiment)

# ----------------------------
# 5. Visualize Earnings Data
# ----------------------------
def visualize_earnings_data(earnings_df, ticker):
    """
    Visualize earnings data for a specific ticker.
    """
    if earnings_df is not None and 'fiscalDateEnding' in earnings_df.columns:
        # Filter earnings data for the specific ticker
        ticker_earnings = earnings_df[earnings_df['ticker'] == ticker]
        
        # Plot reported EPS vs estimated EPS
        plt.figure(figsize=(12, 6))
        plt.plot(ticker_earnings['fiscalDateEnding'], ticker_earnings['reportedEPS'], marker='o', label='Reported EPS')
        plt.plot(ticker_earnings['fiscalDateEnding'], ticker_earnings['estimatedEPS'], marker='o', label='Estimated EPS')
        plt.title(f"Reported vs Estimated EPS for {ticker}")
        plt.xlabel("Fiscal Date Ending")
        plt.ylabel("EPS ($)")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"Warning: Earnings data is missing or does not contain 'fiscalDateEnding' column. Skipping earnings visualization for {ticker}.")

# Example: Visualize earnings data for Apple (AAPL)
visualize_earnings_data(preprocessed_earnings_df, "AAPL")

# ----------------------------
# 6. Additional Charts for Earnings Data
# ----------------------------
def visualize_earnings_analysis(earnings_df):
    """
    Visualize additional earnings analysis charts.
    """
    if earnings_df is not None and 'surprisePercentage' in earnings_df.columns:
        # Surprise percentage distribution
        plt.figure(figsize=(10, 6))
        plt.hist(earnings_df['surprisePercentage'], bins=50, color='orange', alpha=0.7)
        plt.title("Distribution of Earnings Surprise Percentage")
        plt.xlabel("Surprise Percentage")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

        # Earnings trends over time
        earnings_df['year'] = pd.to_datetime(earnings_df['fiscalDateEnding']).dt.year
        yearly_earnings = earnings_df.groupby('year')['reportedEPS'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_earnings['year'], yearly_earnings['reportedEPS'], marker='o', color='red')
        plt.title("Average Reported EPS Over Time")
        plt.xlabel("Year")
        plt.ylabel("Reported EPS ($)")
        plt.grid()
        plt.show()
    else:
        print("Warning: Earnings data is missing or does not contain required columns. Skipping earnings analysis visualization.")

# Visualize additional earnings analysis
visualize_earnings_analysis(preprocessed_earnings_df)

# ----------------------------
# 7. Visualize Macroeconomic Data
# ----------------------------
def visualize_macroeconomic_data(fred_df):
    """
    Visualize macroeconomic data (e.g., GDP).
    """
    if fred_df is not None and 'GDP' in fred_df.columns:
        # Plot GDP over time
        plt.figure(figsize=(12, 6))
        plt.plot(fred_df.index, fred_df['GDP'], marker='o', color='red')
        plt.title("Quarterly GDP Over Time")
        plt.xlabel("Date")
        plt.ylabel("GDP ($)")
        plt.grid()
        plt.show()

        # Calculate and plot GDP growth rate
        fred_df['GDP Growth Rate'] = fred_df['GDP'].pct_change() * 100
        plt.figure(figsize=(12, 6))
        plt.plot(fred_df.index, fred_df['GDP Growth Rate'], marker='o', color='blue')
        plt.title("Quarterly GDP Growth Rate Over Time")
        plt.xlabel("Date")
        plt.ylabel("GDP Growth Rate (%)")
        plt.grid()
        plt.show()
    else:
        print("Warning: Macroeconomic data is missing or does not contain 'GDP' column. Skipping GDP visualization.")

# Visualize GDP data
visualize_macroeconomic_data(preprocessed_fred_df)

# ----------------------------
# 8. Interactive Plotly Chart for Cumulative Returns
# ----------------------------
def visualize_interactive_cumulative_returns(stock_df):
    """
    Visualize cumulative returns using an interactive Plotly chart.
    """
    if stock_df is not None and len(stock_df.columns) > 0:
        # Calculate cumulative returns
        cumulative_returns = (1 + stock_df.xs('Close', axis=1, level=1).pct_change()).cumprod()

        # Melt the DataFrame for Plotly
        cumulative_returns_melted = cumulative_returns.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Cumulative Return')

        # Create an interactive line plot
        fig = px.line(cumulative_returns_melted, x='Date', y='Cumulative Return', color='Ticker',
                      title="Cumulative Returns for S&P 500 Stocks",
                      height=800,  # Increase chart height
                      labels={'Cumulative Return': 'Cumulative Return (Log Scale)'},
                      log_y=True)  # Use logarithmic scale for y-axis

        # Update layout for better spacing
        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=50),  # Add margins
            legend=dict(
                title='Ticker',
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                font=dict(size=10),  # Adjust legend font size
            ),
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Log Scale)",
            hovermode="x unified",  # Show hover information for all lines at once
        )

        # Enable scroll zoom
        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=True)),  # Add a range slider for zooming
        )

        # Add a legend scrollbar (if needed)
        fig.update_layout(
            legend=dict(
                itemsizing='constant',  # Keep legend item sizes consistent
                tracegroupgap=5,  # Add space between legend items
            )
        )

        # Show the plot
        fig.show()
    else:
        print("Warning: Stock data is missing or empty. Skipping interactive cumulative returns visualization.")

# Visualize interactive cumulative returns
visualize_interactive_cumulative_returns(preprocessed_stock_df)

# ----------------------------
# Summary
# ----------------------------
print("\nData visualization and analysis complete.")