import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

class DataFetcher:
    def __init__(self):
        self.api = tradeapi.REST(
            ALPACA_API_KEY, 
            ALPACA_SECRET_KEY, 
            ALPACA_BASE_URL,
            api_version='v2'
        )
    
    def get_historical_data(self, symbols, start_date=None, end_date=None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        price_data = {}
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'MATIC', 'LINK', 'DOT', 'UNI', 'AVAX', 
                         'BNB', 'XRP', 'ATOM', 'LTC', 'BCH', 'ALGO', 'XLM', 'VET', 'FIL', 'AAVE']
        
        for symbol in symbols:
            try:
                # Clean symbol (remove any trailing spaces or special characters)
                clean_symbol = symbol.strip().upper()
                
                # More specific crypto detection
                # Check for explicit crypto patterns or known crypto symbols
                base_symbol = clean_symbol.replace('-USD', '').replace('/USD', '').replace('USD', '') if 'USD' in clean_symbol else clean_symbol
                is_crypto = (base_symbol in crypto_symbols or 
                           ('/' in clean_symbol and 'USD' in clean_symbol) or 
                           ('-USD' in clean_symbol and base_symbol in crypto_symbols) or
                           (clean_symbol.endswith('USD') and len(base_symbol) <= 5 and base_symbol in crypto_symbols))
                
                if is_crypto:
                    # Format crypto symbol properly for Alpaca API (needs to be XXX/USD format)
                    if '-USD' in clean_symbol:
                        # Convert BTC-USD to BTC/USD
                        crypto_symbol = clean_symbol.replace('-USD', '/USD')
                    elif '-' in clean_symbol and not clean_symbol.endswith('USD'):
                        # Convert BTC-USDT or similar to BTC/USD
                        base_symbol = clean_symbol.split('-')[0]
                        crypto_symbol = f"{base_symbol}/USD"
                    elif '/' not in clean_symbol and not clean_symbol.endswith('USD'):
                        # Convert BTC to BTC/USD
                        crypto_symbol = f"{clean_symbol}/USD"
                    elif clean_symbol.endswith('USD') and '/' not in clean_symbol and '-' not in clean_symbol:
                        # Convert BTCUSD to BTC/USD
                        crypto_symbol = clean_symbol[:-3] + '/USD'
                    elif '/' in clean_symbol:
                        # Already in correct format
                        crypto_symbol = clean_symbol
                    else:
                        crypto_symbol = f"{clean_symbol}/USD"
                    
                    print(f"Fetching crypto data for {crypto_symbol}")
                    bars = self.api.get_crypto_bars(
                        crypto_symbol,
                        tradeapi.rest.TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    ).df
                    
                    if not bars.empty and len(bars) > 0:
                        price_data[clean_symbol] = bars['close']
                        print(f"Successfully fetched {len(bars)} days of crypto data for {clean_symbol}")
                else:
                    # Regular stock symbol
                    bars = self.api.get_bars(
                        clean_symbol,
                        tradeapi.rest.TimeFrame.Day,
                        start=start_date,
                        end=end_date,
                        adjustment='all',
                        feed='iex',  # Use IEX feed for paper trading
                        limit=1000
                    ).df
                    
                    if not bars.empty and len(bars) > 0:
                        price_data[clean_symbol] = bars['close']
                        print(f"Successfully fetched {len(bars)} days of data for {clean_symbol}")
                    else:
                        print(f"No data available for {clean_symbol}")
                        
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        if price_data:
            df = pd.DataFrame(price_data)
            # Forward fill then backward fill to handle any missing data
            df = df.ffill().bfill()
            print(f"Retrieved data for {len(df.columns)} symbols with {len(df)} trading days")
            return df
        
        print("No price data could be fetched")
        return pd.DataFrame()
    
    def get_spy_benchmark(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            bars = self.api.get_bars(
                'SPY',
                tradeapi.rest.TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment='all',
                feed='iex',  # Use IEX feed for paper trading
                limit=1000
            ).df
            
            if not bars.empty:
                print(f"Successfully fetched SPY benchmark data: {len(bars)} days")
                return bars['close']
        except Exception as e:
            print(f"Error fetching SPY data: {str(e)}")
        
        return pd.Series()
    
    def get_crypto_recommendations(self):
        crypto_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOGE/USD']
        crypto_data = {}
        
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        for symbol in crypto_symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol,
                    tradeapi.rest.TimeFrame.Day,
                    start=start_date,
                    end=end_date
                ).df
                
                if not bars.empty and len(bars) > 20:
                    returns = bars['close'].pct_change().dropna()
                    display_symbol = symbol.replace('/USD', '')
                    crypto_data[display_symbol] = {
                        'return': returns.mean() * 252,
                        'volatility': returns.std() * np.sqrt(252),
                        'sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                        'momentum': (bars['close'].iloc[-1] / bars['close'].iloc[0] - 1) if len(bars) > 0 else 0
                    }
            except Exception as e:
                print(f"Error fetching crypto data for {symbol}: {e}")
                continue
        
        return crypto_data
    
    def get_stock_recommendations(self, current_symbols, sector_diversity=True):
        potential_stocks = [
            'QQQ', 'IWM', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI',
            'MA', 'UNH', 'BRK.B', 'AVGO', 'LLY', 'COST', 'ABBV', 'CRM', 'AMD', 'ADBE'
        ]
        
        potential_stocks = [s for s in potential_stocks if s not in current_symbols]
        
        stock_metrics = {}
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        for symbol in potential_stocks:
            try:
                bars = self.api.get_bars(
                    symbol,
                    tradeapi.rest.TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                    adjustment='all',
                    feed='iex',
                    limit=100
                ).df
                
                if not bars.empty and len(bars) > 20:
                    returns = bars['close'].pct_change().dropna()
                    stock_metrics[symbol] = {
                        'return': returns.mean() * 252,
                        'volatility': returns.std() * np.sqrt(252),
                        'sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                        'momentum': (bars['close'].iloc[-1] / bars['close'].iloc[0] - 1)
                    }
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return stock_metrics