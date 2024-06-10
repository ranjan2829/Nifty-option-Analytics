import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from requests.exceptions import RequestException, JSONDecodeError

# Function to fetch data from the NSE option chain site
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

def fetch_live_nifty_data():
    url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    session = requests.Session()
    session.get('https://www.nseindia.com', headers=headers)  # Initial request to set the cookie
    
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

def plot_option_data(calls_df, puts_df, strikes_range):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Plot Change in Open Interest (COI)
    fig.add_trace(go.Scatter(x=calls_df['strikePrice'], y=calls_df['changeInOpenInterest'], mode='lines+markers', name='Call Options COI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=puts_df['strikePrice'], y=puts_df['changeInOpenInterest'], mode='lines+markers', name='Put Options COI'), row=1, col=1)

    # Plot Total Traded Volume
    fig.add_trace(go.Scatter(x=calls_df['strikePrice'], y=calls_df['totalTradedVolume'], mode='lines+markers', name='Call Options Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=puts_df['strikePrice'], y=puts_df['totalTradedVolume'], mode='lines+markers', name='Put Options Volume'), row=2, col=1)

    # Plot Implied Volatility
    fig.add_trace(go.Scatter(x=calls_df['strikePrice'], y=calls_df['impliedVolatility'], mode='lines+markers', name='Call Options IV'), row=3, col=1)
    fig.add_trace(go.Scatter(x=puts_df['strikePrice'], y=puts_df['impliedVolatility'], mode='lines+markers', name='Put Options IV'), row=3, col=1)

    # Plot Last Price
    fig.add_trace(go.Scatter(x=calls_df['strikePrice'], y=calls_df['lastPrice'], mode='lines+markers', name='Call Options Last Price'), row=4, col=1)
    fig.add_trace(go.Scatter(x=puts_df['strikePrice'], y=puts_df['lastPrice'], mode='lines+markers', name='Put Options Last Price'), row=4, col=1)

    # Plot Change in Price
    fig.add_trace(go.Scatter(x=calls_df['strikePrice'], y=calls_df['change'], mode='lines+markers', name='Call Options Change in Price'), row=5, col=1)
    fig.add_trace(go.Scatter(x=puts_df['strikePrice'], y=puts_df['change'], mode='lines+markers', name='Put Options Change in Price'), row=5, col=1)

    fig.update_layout(height=2000, width=800, title_text="NIFTY Options Data")
    
    st.plotly_chart(fig)

def main():
    # Streamlit UI
    st.title("NIFTY Options Data and Live Chart")

    if st.button("Refresh Options Data"):
        data = fetch_option_chain_data()
        if data is None:
            st.error("Failed to fetch data after multiple retries.")
            return

        underlying_value = data['records']['underlyingValue']
        call_data = []
        put_data = []

        for record in data['records']['data']:
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

        # Step 3: Find the ATM strike price
        calls_df['diff'] = abs(calls_df['strikePrice'] - underlying_value)
        puts_df['diff'] = abs(puts_df['strikePrice'] - underlying_value)

        atm_strike = calls_df.loc[calls_df['diff'].idxmin(), 'strikePrice']

        calls_df = calls_df.drop(columns=['diff'])
        puts_df = puts_df.drop(columns=['diff'])

        unique_strikes = sorted(calls_df['strikePrice'].unique())

        atm_index = unique_strikes.index(atm_strike)

        # Calculate the range of strike prices (5 steps up and down)
        start_index = max(atm_index - 5, 0)
        end_index = min(atm_index + 5 + 1, len(unique_strikes))
        strikes_range = unique_strikes[start_index:end_index]

        # Filter the DataFrames for the range of strike prices
        calls_range_df = calls_df[calls_df['strikePrice'].isin(strikes_range)]
        puts_range_df = puts_df[puts_df['strikePrice'].isin(strikes_range)]

        # Plotting the data
        plot_option_data(calls_range_df, puts_range_df, strikes_range)

    if st.button("Show Live NIFTY Chart"):
        nifty_data = fetch_live_nifty_data()
        if nifty_data is None:
            st.error("Failed to fetch live NIFTY data.")
            return

        nifty_values = []
        nifty_timestamps = []
        for record in nifty_data['data']:
            nifty_values.append(record['last'])
            nifty_timestamps.append(record['timestamp'])

        fig = go.Figure(data=go.Scatter(x=nifty_timestamps, y=nifty_values, mode='lines+markers', name='NIFTY Live'))

        fig.update_layout(title='NIFTY Live Chart',
                          xaxis_title='Time',
                          yaxis_title='Index Value')

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
