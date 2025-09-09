import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

print("Testing Alpaca Crypto API...")

api = tradeapi.REST(
    ALPACA_API_KEY, 
    ALPACA_SECRET_KEY, 
    ALPACA_BASE_URL,
    api_version='v2'
)

# Test different crypto symbol formats
test_symbols = [
    'BTC/USD',      # Correct format
    'ETH/USD',      # Correct format
    'SOL/USD',      # Correct format
]

start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

for symbol in test_symbols:
    print(f"\nTesting {symbol}...")
    try:
        bars = api.get_crypto_bars(
            symbol,
            tradeapi.rest.TimeFrame.Day,
            start=start_date,
            end=end_date
        ).df
        
        if not bars.empty:
            print(f"✓ Success: Got {len(bars)} days of data")
            print(f"  Latest price: ${bars['close'].iloc[-1]:.2f}")
            print(f"  Date range: {bars.index[0]} to {bars.index[-1]}")
        else:
            print(f"✗ No data returned")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

# Now test the conversion logic
print("\n" + "="*50)
print("Testing symbol conversion logic:")

test_conversions = [
    'BTC-USD',      # Should convert to BTC/USD
    'ETH-USD',      # Should convert to ETH/USD
    'BTCUSD',       # Should convert to BTC/USD
    'BTC',          # Should convert to BTC/USD
    'SOL-USD',      # Should convert to SOL/USD
]

for original in test_conversions:
    # Apply the conversion logic
    clean_symbol = original.strip().upper()
    
    if '-USD' in clean_symbol:
        converted = clean_symbol.replace('-USD', '/USD')
    elif clean_symbol.endswith('USD') and '/' not in clean_symbol and '-' not in clean_symbol:
        converted = clean_symbol[:-3] + '/USD'
    elif '/' not in clean_symbol and not clean_symbol.endswith('USD'):
        converted = f"{clean_symbol}/USD"
    else:
        converted = clean_symbol
    
    print(f"{original:12} -> {converted:12}", end="")
    
    # Test if it works
    try:
        bars = api.get_crypto_bars(
            converted,
            tradeapi.rest.TimeFrame.Day,
            start=start_date,
            end=end_date,
            limit=1
        ).df
        
        if not bars.empty:
            print(" ✓ Works!")
        else:
            print(" ✗ No data")
    except Exception as e:
        print(f" ✗ Error: {str(e)[:50]}")