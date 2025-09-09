import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

# Test Alpaca API connection
print("Testing Alpaca API connection...")
print(f"API Key: {ALPACA_API_KEY[:10]}...")
print(f"Base URL: {ALPACA_BASE_URL}")

try:
    api = tradeapi.REST(
        ALPACA_API_KEY, 
        ALPACA_SECRET_KEY, 
        ALPACA_BASE_URL,
        api_version='v2'
    )
    
    # Test account access
    account = api.get_account()
    print(f"\n✓ Successfully connected to Alpaca API")
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${account.buying_power}")
    
    # Test fetching data for a single stock
    test_symbol = 'AAPL'
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\nFetching data for {test_symbol} from {start_date} to {end_date}...")
    
    bars = api.get_bars(
        test_symbol,
        tradeapi.rest.TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment='all',
        feed='iex',  # Use IEX feed for paper trading
        limit=100
    ).df
    
    if not bars.empty:
        print(f"✓ Successfully fetched {len(bars)} days of data for {test_symbol}")
        print(f"Latest price: ${bars['close'].iloc[-1]:.2f}")
        print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
    else:
        print(f"✗ No data returned for {test_symbol}")
        
except Exception as e:
    print(f"\n✗ Error connecting to Alpaca API: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check that your API keys are correct")
    print("2. Ensure you're using paper trading keys for the paper trading URL")
    print("3. Check your internet connection")
    print("4. Verify the API endpoint is correct")