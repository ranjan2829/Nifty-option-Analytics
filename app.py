import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from requests.exceptions import RequestException, JSONDecodeError
from bs4 import BeautifulSoup

def fetch_option_chain_data():
    url = 'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    session = requests.Session()
    session.get('https://www.nseindia.com', headers=headers)  # Initial request to set the cookie
    
    max_retries = 3
    for _ in range(max_retries):
        try:
            response = session.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()
            return data
        except JSONDecodeError:
            st.error("Error: Unable to parse JSON response.")
        except RequestException as e:
            st.error(f"Request error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    
    return None

def fetch_live_market_data():
    url = 'https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%2050'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.text
    except requests.RequestException as e:
        st.error(f"Request error: {e}")
        return None

def parse_live_market_data(html_content):
    if html_content is None:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table', {'id': 'equity_stockIndices_data_table'})

    if table is None:
        st.error("Failed to find live market data.")
        return None

    headers = [th.text for th in table.find_all('th')]
    rows = []

    for tr in table.find_all('tr')[1:]:
        row = [td.text.strip() for td in tr.find_all('td')]
        rows.append(row)

    return headers, rows

def plot_heatmap(data):
    df = pd.DataFrame(data)
    fig = px.imshow(df)
    st.plotly_chart(fig)

def plot_option_chain(option_chain_data):
    underlying_value = option_chain_data['records']['underlyingValue']
    call_data = []
    put_data = []

    for record in option_chain_data['records']['data']:
        if 'CE' in record and 'PE' in record:
            call_data.append({
                'strikePrice': record['strikePrice'],
                'impliedVolatility': record['CE']['impliedVolatility'],
                'openInterest': record['CE']['openInterest'],
                'changeInOpenInterest': record['CE']['changeinOpenInterest'],
                'totalTradedVolume': record['CE']['totalTradedVolume'],
                'lastPrice': record['CE']['lastPrice'],
                'change': record['CE']['change']
            })
            put_data.append({
                'strikePrice': record['strikePrice'],
                'impliedVolatility': record['PE']['impliedVolatility'],
                'openInterest': record['PE']['openInterest'],
                'changeInOpenInterest': record['PE']['changeinOpenInterest'],
                'totalTradedVolume': record['PE']['totalTradedVolume'],
                'lastPrice': record['PE']['lastPrice'],
                'change': record['PE']['change']
            })

    calls_df = pd.DataFrame(call_data)
    puts_df = pd.DataFrame(put_data)

    calls_df['diff'] = abs(calls_df['strikePrice'] - underlying_value)
    puts_df['diff'] = abs(puts_df['strikePrice'] - underlying_value)

    atm_strike = calls_df.loc[calls_df['diff'].idxmin(), 'strikePrice']

    calls_df = calls_df.drop(columns=['diff'])
    puts_df = puts_df.drop(columns=['diff'])

    unique_strikes = sorted(calls_df['strikePrice'].unique())

    atm_index = unique_strikes.index(atm_strike)

    start_index = max(atm_index - 5, 0)
    end_index = min(atm_index + 5 + 1, len(unique_strikes))
    strikes_range = unique_strikes[start_index:end_index]

    calls_range_df = calls_df[calls_df['strikePrice'].isin(strikes_range)]
    puts_range_df = puts_df[puts_df['strikePrice'].isin(strikes_range)]

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=calls_range_df['strikePrice'], y=calls_range_df['changeInOpenInterest'], mode='lines+markers', name='Call Options COI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=puts_range_df['strikePrice'], y=puts_range_df['changeInOpenInterest'], mode='lines+markers', name='Put Options COI'), row=1, col=1)

    fig.add_trace(go.Scatter(x=calls_range_df['strikePrice'], y=calls_range_df['totalTradedVolume'], mode='lines+markers', name='Call Options Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=puts_range_df['strikePrice'], y=puts_range_df['totalTradedVolume'], mode='lines+markers', name='Put Options Volume'), row=2, col=1)

    fig.add_trace(go.Scatter(x=calls_range_df['strikePrice'], y=calls_range_df['impliedVolatility'], mode='lines+markers', name='Call Options IV'), row=3, col=1)
    fig.add_trace(go.Scatter(x=puts_range_df['strikePrice'], y=puts_range_df['impliedVolatility'], mode='lines+markers', name='Put Options IV'), row=3, col=1)

    fig.add_trace(go.Scatter(x=calls_range_df['strikePrice'], y=calls_range_df['lastPrice'], mode='lines+markers', name='Call Options Last Price'), row=4, col=1)
    fig.add_trace(go.Scatter(x=puts_range_df['strikePrice'], y=puts_range_df['lastPrice'], mode='lines+markers', name='Put Options Last Price'), row=4, col=1)

    fig.add_trace(go.Scatter(x=calls_range_df['strikePrice'], y=calls_range_df['change'], mode='lines+markers', name='Call Options Change in Price'), row=5, col=1)
    fig.add_trace(go.Scatter(x=puts_range_df['strikePrice'], y=puts_range_df['change'], mode='lines+markers', name='Put Options Change in Price'), row=5, col=1)

    fig.update_layout(height=2000, width=800, title_text="NIFTY Options Data")
    
    st.plotly_chart(fig)

def main():
    st.title("NIFTY Options Data and Live Market Update")

    if st.button("Refresh Options Data"):
        option_chain_data = fetch_option_chain_data()
        if option_chain_data is None:
            st.error("Failed to fetch option chain data after multiple retries.")
            return
        plot_option_chain(option_chain_data)

    if st.button("Refresh Live Market Data"):
        live_market_data = fetch_live_market_data()
        if live_market_data is None:
            st.error("Failed to fetch live market data.")
            return
        headers, rows = parse_live_market_data(live_market_data)
        if headers and rows:
            st.table(rows, headers=headers)
        else:
            st.error("Failed to fetch or parse live market data.")

if __name__ == '__main__':
    main()
