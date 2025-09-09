import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class PortfolioOptimizer:
    def __init__(self, price_data):
        self.price_data = price_data
        self.returns = price_data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def calculate_portfolio_metrics(self, weights, returns=None):
        if returns is None:
            returns = self.returns
        
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe': sharpe_ratio
        }
    
    def optimize_sharpe_ratio(self, current_weights=None):
        n_assets = len(self.returns.columns)
        
        if current_weights is None:
            initial_weights = np.array([1/n_assets] * n_assets)
        else:
            initial_weights = current_weights
        
        def neg_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return -metrics['sharpe']
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = minimize(neg_sharpe, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def ml_optimization(self, lookback_periods=60):
        # Adjust lookback period based on available data
        available_periods = len(self.returns)
        if available_periods < lookback_periods + 20:
            # If we don't have enough data, use a simpler approach
            lookback_periods = max(20, available_periods // 3)
        
        if available_periods < 30:
            # Not enough data for ML, fall back to traditional optimization
            print(f"Insufficient data for ML optimization ({available_periods} days). Using traditional optimization.")
            return self.optimize_sharpe_ratio()
        
        feature_data = []
        target_data = []
        
        for i in range(lookback_periods, len(self.returns)):
            window_returns = self.returns.iloc[i-lookback_periods:i]
            
            features = []
            for col in window_returns.columns:
                col_returns = window_returns[col]
                # Handle edge cases where data might be constant
                features.extend([
                    col_returns.mean(),
                    col_returns.std() if col_returns.std() > 0 else 0.001,
                    col_returns.skew() if len(col_returns) > 2 else 0,
                    col_returns.iloc[-1],
                    (col_returns.iloc[-1] / col_returns.iloc[0]) - 1 if col_returns.iloc[0] != 0 else 0
                ])
            
            feature_data.append(features)
            
            # Calculate target sharpe ratios
            next_returns = self.returns.iloc[i]
            sharpe_values = []
            for j, asset in enumerate(self.returns.columns):
                weight_vector = np.zeros(len(self.returns.columns))
                weight_vector[j] = 1
                window_for_sharpe = self.returns.iloc[max(0, i-20):min(i+1, len(self.returns))]
                if len(window_for_sharpe) > 1:
                    metrics = self.calculate_portfolio_metrics(weight_vector, window_for_sharpe)
                    sharpe_values.append(metrics['sharpe'])
                else:
                    sharpe_values.append(0)
            
            target_data.append(sharpe_values)
        
        if not feature_data or len(feature_data) < 5:
            print("Insufficient training data for ML. Using traditional optimization.")
            return self.optimize_sharpe_ratio()
        
        X = np.array(feature_data)
        y = np.array(target_data)
        
        # Check if we have valid data
        if X.shape[0] == 0 or np.isnan(X).any() or np.isinf(X).any():
            print("Invalid feature data. Using traditional optimization.")
            return self.optimize_sharpe_ratio()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Prepare latest features for prediction
        latest_window = self.returns.iloc[-lookback_periods:]
        latest_features = []
        for col in latest_window.columns:
            col_returns = latest_window[col]
            latest_features.extend([
                col_returns.mean(),
                col_returns.std() if col_returns.std() > 0 else 0.001,
                col_returns.skew() if len(col_returns) > 2 else 0,
                col_returns.iloc[-1],
                (col_returns.iloc[-1] / col_returns.iloc[0]) - 1 if col_returns.iloc[0] != 0 else 0
            ])
        
        latest_features = np.array(latest_features).reshape(1, -1)
        
        # Check for invalid features
        if np.isnan(latest_features).any() or np.isinf(latest_features).any():
            print("Invalid latest features. Using traditional optimization.")
            return self.optimize_sharpe_ratio()
        
        latest_features_scaled = scaler.transform(latest_features)
        
        predicted_sharpes = model.predict(latest_features_scaled)[0]
        
        # Create base weights from predictions
        base_weights = np.maximum(predicted_sharpes, 0)
        if base_weights.sum() > 0:
            base_weights = base_weights / base_weights.sum()
        else:
            base_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        
        # Fine-tune with traditional optimization
        optimized_weights = self.optimize_sharpe_ratio(base_weights)
        
        return optimized_weights
    
    def calculate_benchmark_comparison(self, portfolio_weights, spy_prices):
        portfolio_returns = (self.returns * portfolio_weights).sum(axis=1)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        
        spy_returns = spy_prices.pct_change().dropna()
        spy_cumulative = (1 + spy_returns).cumprod()
        
        common_dates = portfolio_cumulative.index.intersection(spy_cumulative.index)
        
        if len(common_dates) > 0:
            portfolio_total_return = (portfolio_cumulative.loc[common_dates].iloc[-1] - 1) * 100
            spy_total_return = (spy_cumulative.loc[common_dates].iloc[-1] - 1) * 100
            
            excess_return = portfolio_total_return - spy_total_return
            
            return {
                'portfolio_return': portfolio_total_return,
                'spy_return': spy_total_return,
                'excess_return': excess_return
            }
        
        return {
            'portfolio_return': 0,
            'spy_return': 0,
            'excess_return': 0
        }