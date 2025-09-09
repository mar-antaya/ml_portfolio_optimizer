import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from portfolio_optimizer import PortfolioOptimizer

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("🚀 AI-Powered Portfolio Optimizer")
st.markdown("Optimize your investment portfolio using machine learning and get personalized recommendations")

if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Your Portfolio")
    
    input_method = st.radio("Choose input method:", ["CSV Upload", "Manual Entry", "Use Example Portfolio"])
    
    if input_method == "CSV Upload":
        uploaded_file = st.file_uploader("Upload your portfolio CSV", type=['csv'])
        if uploaded_file is not None:
            portfolio_df = pd.read_csv(uploaded_file)
            st.session_state.portfolio_data = portfolio_df
    
    elif input_method == "Manual Entry":
        st.write("Enter your portfolio holdings:")
        
        num_holdings = st.number_input("Number of holdings:", min_value=1, max_value=50, value=5)
        
        holdings_data = []
        cols = st.columns(4)
        
        for i in range(num_holdings):
            with cols[i % 4]:
                symbol = st.text_input(f"Symbol {i+1}", key=f"symbol_{i}")
                shares = st.number_input(f"Shares {i+1}", min_value=0.0, key=f"shares_{i}")
                if symbol and shares > 0:
                    holdings_data.append({'Symbol': symbol, 'Shares': shares})
        
        if st.button("Create Portfolio"):
            if holdings_data:
                portfolio_df = pd.DataFrame(holdings_data)
                st.session_state.portfolio_data = portfolio_df
    
    else:  
        example_data = """Symbol,Company Name,Sector,Shares,Market Value,Allocation %
AAPL,Apple Inc.,Technology,100,23790.00,20.0
MSFT,Microsoft Corporation,Technology,50,21450.00,18.0
GOOGL,Alphabet Inc.,Technology,75,14250.00,12.0
AMZN,Amazon.com Inc.,Consumer Discretionary,60,13200.00,11.0
TSLA,Tesla Inc.,Consumer Discretionary,30,11970.00,10.0
META,Meta Platforms Inc.,Technology,25,15000.00,12.6
JPM,JPMorgan Chase & Co.,Financial Services,40,9600.00,8.1
JNJ,Johnson & Johnson,Healthcare,50,9900.00,8.3"""
        
        portfolio_df = pd.read_csv(StringIO(example_data))
        st.session_state.portfolio_data = portfolio_df
        st.info("Using example portfolio data")

with col2:
    st.subheader("⚙️ Optimization Settings")
    
    optimization_method = st.selectbox(
        "Optimization Method:",
        ["ML-Enhanced Optimization", "Traditional Sharpe Optimization"]
    )
    
    include_crypto = st.checkbox("Include Cryptocurrency Recommendations", value=True)
    include_new_stocks = st.checkbox("Include New Stock Recommendations", value=True)
    
    date_range = st.select_slider(
        "Historical Data Period:",
        options=["3 Months", "6 Months", "1 Year", "2 Years"],
        value="1 Year"
    )

if st.session_state.portfolio_data is not None:
    st.subheader("Current Portfolio")
    st.dataframe(st.session_state.portfolio_data)
    
    if st.button("🎯 Optimize Portfolio", type="primary"):
        # Create an expander for console output
        with st.expander("📋 Processing Log", expanded=True):
            log_container = st.empty()
            logs = []
            
        with st.spinner("Fetching market data and optimizing portfolio..."):
            
            fetcher = DataFetcher()
            
            symbols = st.session_state.portfolio_data['Symbol'].tolist()
            logs.append(f"Portfolio symbols: {', '.join(symbols)}")
            log_container.text('\n'.join(logs))
            
            date_mapping = {
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365,
                "2 Years": 730
            }
            days_back = date_mapping[date_range]
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            logs.append(f"Fetching data from {start_date} to {end_date}")
            log_container.text('\n'.join(logs))
            
            price_data = fetcher.get_historical_data(symbols, start_date=start_date)
            
            if price_data.empty:
                st.error("Unable to fetch price data. Please check your symbols and try again.")
                st.info("Note: The free Alpaca paper trading account uses IEX data which may have limited coverage for some symbols. Try using major stocks like AAPL, MSFT, GOOGL, etc.")
            else:
                # Get the symbols that actually have data
                available_symbols = list(price_data.columns)
                logs.append(f"✓ Successfully fetched price data for {len(available_symbols)} symbols: {', '.join(available_symbols)}")
                
                # Find which symbols failed
                failed_symbols = [s for s in symbols if s not in available_symbols]
                if failed_symbols:
                    logs.append(f"⚠️ No data available for: {', '.join(failed_symbols)}")
                    st.warning(f"Could not fetch data for: {', '.join(failed_symbols)}. These will be excluded from optimization.")
                
                log_container.text('\n'.join(logs))
                
                spy_data = fetcher.get_spy_benchmark(start_date=start_date)
                
                optimizer = PortfolioOptimizer(price_data)
                
                # Align weights with available symbols
                portfolio_df = st.session_state.portfolio_data
                if 'Allocation %' in portfolio_df.columns:
                    # Filter to only include symbols with data
                    filtered_df = portfolio_df[portfolio_df['Symbol'].isin(available_symbols)]
                    current_weights = filtered_df['Allocation %'].values / 100
                    # Renormalize weights to sum to 1
                    current_weights = current_weights / current_weights.sum()
                else:
                    # Filter to only include symbols with data
                    filtered_df = portfolio_df[portfolio_df['Symbol'].isin(available_symbols)]
                    market_values = filtered_df['Market Value'] if 'Market Value' in filtered_df.columns else filtered_df['Shares']
                    total_value = market_values.sum()
                    current_weights = market_values.values / total_value
                
                # Update symbols list to match available data
                symbols = available_symbols
                
                current_metrics = optimizer.calculate_portfolio_metrics(current_weights)
                
                if optimization_method == "ML-Enhanced Optimization":
                    optimized_weights = optimizer.ml_optimization()
                else:
                    optimized_weights = optimizer.optimize_sharpe_ratio()
                
                optimized_metrics = optimizer.calculate_portfolio_metrics(optimized_weights)
                
                benchmark_comparison = optimizer.calculate_benchmark_comparison(current_weights, spy_data)
                optimized_benchmark = optimizer.calculate_benchmark_comparison(optimized_weights, spy_data)
                
                recommendations = []
                if include_new_stocks:
                    stock_recs = fetcher.get_stock_recommendations(symbols)
                    top_stocks = sorted(stock_recs.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:5]
                    for symbol, metrics in top_stocks:
                        recommendations.append({
                            'Symbol': symbol,
                            'Type': 'Stock',
                            'Expected Return': f"{metrics['return']*100:.2f}%",
                            'Sharpe Ratio': f"{metrics['sharpe']:.2f}",
                            'Momentum': f"{metrics['momentum']*100:.2f}%"
                        })
                
                if include_crypto:
                    crypto_recs = fetcher.get_crypto_recommendations()
                    top_crypto = sorted(crypto_recs.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:3]
                    for symbol, metrics in top_crypto:
                        recommendations.append({
                            'Symbol': symbol,
                            'Type': 'Crypto',
                            'Expected Return': f"{metrics['return']*100:.2f}%",
                            'Sharpe Ratio': f"{metrics['sharpe']:.2f}",
                            'Momentum': f"{metrics.get('momentum', 0)*100:.2f}%" if 'momentum' in metrics else '-'
                        })
                
                st.session_state.optimization_results = {
                    'current_metrics': current_metrics,
                    'optimized_metrics': optimized_metrics,
                    'current_weights': current_weights,
                    'optimized_weights': optimized_weights,
                    'symbols': symbols,
                    'benchmark_comparison': benchmark_comparison,
                    'optimized_benchmark': optimized_benchmark,
                    'recommendations': recommendations
                }

if st.session_state.optimization_results:
    results = st.session_state.optimization_results
    
    st.markdown("---")
    st.header("📈 Optimization Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Sharpe Ratio",
            f"{results['current_metrics']['sharpe']:.3f}",
        )
        st.metric(
            "Optimized Sharpe Ratio",
            f"{results['optimized_metrics']['sharpe']:.3f}",
            delta=f"{(results['optimized_metrics']['sharpe'] - results['current_metrics']['sharpe']):.3f}"
        )
    
    with col2:
        st.metric(
            "Current Annual Return",
            f"{results['current_metrics']['return']*100:.2f}%",
        )
        st.metric(
            "Optimized Annual Return",
            f"{results['optimized_metrics']['return']*100:.2f}%",
            delta=f"{(results['optimized_metrics']['return'] - results['current_metrics']['return'])*100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Current Volatility",
            f"{results['current_metrics']['volatility']*100:.2f}%",
        )
        st.metric(
            "Optimized Volatility",
            f"{results['optimized_metrics']['volatility']*100:.2f}%",
            delta=f"{(results['optimized_metrics']['volatility'] - results['current_metrics']['volatility'])*100:.2f}%"
        )
    
    with col4:
        st.metric(
            "vs SPY (Current)",
            f"{results['benchmark_comparison']['excess_return']:.2f}%",
        )
        st.metric(
            "vs SPY (Optimized)",
            f"{results['optimized_benchmark']['excess_return']:.2f}%",
            delta=f"{(results['optimized_benchmark']['excess_return'] - results['benchmark_comparison']['excess_return']):.2f}%"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current vs Optimized Weights")
        
        weights_df = pd.DataFrame({
            'Symbol': results['symbols'],
            'Current Weight (%)': results['current_weights'] * 100,
            'Optimized Weight (%)': results['optimized_weights'] * 100,
            'Change (%)': (results['optimized_weights'] - results['current_weights']) * 100
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current', x=weights_df['Symbol'], y=weights_df['Current Weight (%)']))
        fig.add_trace(go.Bar(name='Optimized', x=weights_df['Symbol'], y=weights_df['Optimized Weight (%)']))
        fig.update_layout(barmode='group', title='Portfolio Weight Comparison', xaxis_title='Symbol', yaxis_title='Weight (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(weights_df.style.format({
            'Current Weight (%)': '{:.2f}',
            'Optimized Weight (%)': '{:.2f}',
            'Change (%)': '{:+.2f}'
        }))
    
    with col2:
        st.subheader("Portfolio Allocation Pie Charts")
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=results['symbols'],
            values=results['current_weights'],
            name="Current",
            domain={'x': [0, 0.45], 'y': [0, 1]},
            title='Current'
        ))
        fig.add_trace(go.Pie(
            labels=results['symbols'],
            values=results['optimized_weights'],
            name="Optimized",
            domain={'x': [0.55, 1], 'y': [0, 1]},
            title='Optimized'
        ))
        fig.update_layout(title='Portfolio Allocation Comparison', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    if results['recommendations']:
        st.subheader("🎯 Recommended Additions")
        st.write("Consider adding these assets to outperform SPY:")
        
        rec_df = pd.DataFrame(results['recommendations'])
        st.dataframe(rec_df.style.highlight_max(subset=['Sharpe Ratio'], color='lightgreen'))
    
    st.subheader("📊 Risk-Return Profile")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[results['current_metrics']['volatility']*100],
        y=[results['current_metrics']['return']*100],
        mode='markers',
        name='Current Portfolio',
        marker=dict(size=15, color='red', symbol='diamond')
    ))
    
    fig.add_trace(go.Scatter(
        x=[results['optimized_metrics']['volatility']*100],
        y=[results['optimized_metrics']['return']*100],
        mode='markers',
        name='Optimized Portfolio',
        marker=dict(size=15, color='green', symbol='star')
    ))
    
    fig.update_layout(
        title='Risk-Return Comparison',
        xaxis_title='Risk (Volatility %)',
        yaxis_title='Expected Return (%)',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("💡 Investment Insights"):
        st.write(f"""
        ### Key Findings:
        - Your optimized portfolio shows a **{((results['optimized_metrics']['sharpe'] / results['current_metrics']['sharpe']) - 1) * 100:.1f}%** improvement in risk-adjusted returns
        - The optimization suggests {"increasing" if results['optimized_metrics']['return'] > results['current_metrics']['return'] else "decreasing"} overall portfolio risk
        - Your current portfolio {"outperforms" if results['benchmark_comparison']['excess_return'] > 0 else "underperforms"} SPY by **{abs(results['benchmark_comparison']['excess_return']):.2f}%**
        - The optimized portfolio is expected to {"outperform" if results['optimized_benchmark']['excess_return'] > 0 else "underperform"} SPY by **{abs(results['optimized_benchmark']['excess_return']):.2f}%**
        
        ### Recommendations:
        1. Review the suggested weight adjustments and rebalance gradually
        2. Consider the recommended new assets for diversification
        3. Monitor performance quarterly and re-optimize as needed
        """)

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Always consult with a financial advisor before making investment decisions.")