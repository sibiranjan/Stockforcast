import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set page configuration and change website name
st.set_page_config(page_title="Stock Forecast App", page_icon=":chat_with_upwards_trend:")

st.title('Stock Forecast ')


stocks= pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
selected_stock = st.selectbox('Select Ticker', stocks)

stock_info = yf.Ticker(selected_stock).info

def section_header(title):
    st.write(f"## {title}")
    st.markdown("---")
# Displaying basic information
section_header("Basic Information")
st.write(f"**Company Name:** {stock_info.get('longName', 'N/A')}")
st.write(f"**Exchange:** {stock_info.get('exchange', 'N/A')}")
st.write(f"**Sector:** {stock_info.get('sectorDisp', 'N/A')}")
st.write(f"**Industry:** {stock_info.get('industryDisp', 'N/A')}")
st.write(f"**Website:** [{stock_info.get('website', 'N/A')}]({stock_info.get('website', 'N/A')})")
st.write(f"**Address:** {stock_info.get('address1', 'N/A')}")
st.write(f"**Phone:** {stock_info.get('phone', 'N/A')}")
st.write("\n\n")

st.header("**Business Summary**")
string_summary = stock_info.get('longBusinessSummary','N/A')
st.info(string_summary)


# Displaying financial metrics
section_header("Financial Metrics")
st.write(f"**Previous Close Price:** ${stock_info.get('previousClose', 'N/A')}")
st.write(f"**Open Price:** ${stock_info.get('open', 'N/A')}")
st.write(f"**Day's Low:** ${stock_info.get('dayLow', 'N/A')}")
st.write(f"**Day's High:** ${stock_info.get('dayHigh', 'N/A')}")
st.write(f"**52-week Low:** ${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
st.write(f"**52-week High:** ${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
st.write(f"**Market Cap:** ${stock_info.get('marketCap', 'N/A')}")
st.write(f"**Price-to-Earnings Ratio (P/E Ratio):** {stock_info.get('trailingPE', 'N/A')}")
st.write(f"**Earnings per Share (EPS):** {stock_info.get('trailingEps', 'N/A')}")
st.write(f"**Price-to-Book Ratio (P/B Ratio):** {stock_info.get('priceToBook', 'N/A')}")
st.write("\n\n")
# Displaying market metrics
section_header("Market Metrics")
st.write(f"**Volume:** {stock_info.get('volume', 'N/A')}")
st.write(f"**Average Volume:** {stock_info.get('averageVolume', 'N/A')}")
st.write(f"**Bid Price:** ${stock_info.get('bid', 'N/A')}")
st.write(f"**Ask Price:** ${stock_info.get('ask', 'N/A')}")
st.write(f"**Bid Size:** {stock_info.get('bidSize', 'N/A')}")
st.write(f"**Ask Size:** {stock_info.get('askSize', 'N/A')}")
st.write("\n\n")


st.markdown("<h3>Start date</h3>", unsafe_allow_html=True)
START = st.date_input("", datetime.date(2019, 1, 1))

st.markdown("<h3>End date</h3>", unsafe_allow_html=True)
END = st.date_input("", datetime.date(2024, 1, 1))
st.write("\n\n")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Data Loaded successfully!')





st.write("\n\n")
st.header("**Data**")
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


st.markdown("<h3>Years of prediction:</h3>", unsafe_allow_html=True)
n_years = st.slider('', 1, 5)
period = n_years * 365

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
st.write("\n\n")
st.write(f'**Forecast plot for {n_years} years**')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.header("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)



