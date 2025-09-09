import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AdvancedPortfolioOptimizer:
    def __init__(self, price_data, risk_profile='moderate'):
        self.price_data = price_data
        self.returns = price_data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        self.risk_profile = risk_profile
        
        # Risk profile parameters
        self.risk_params = {
            'conservative': {
                'max_volatility': 0.10,  # 10% annual volatility
                'min_sharpe': 0.5,
                'max_single_weight': 0.25,  # Max 25% in single asset
                'risk_free_rate': 0.05
            },
            'moderate': {
                'max_volatility': 0.20,  # 20% annual volatility
                'min_sharpe': 0.75,
                'max_single_weight': 0.35,  # Max 35% in single asset
                'risk_free_rate': 0.05
            },
            'aggressive': {
                'max_volatility': 0.35,  # 35% annual volatility
                'min_sharpe': 1.0,
                'max_single_weight': 0.50,  # Max 50% in single asset
                'risk_free_rate': 0.05
            },
            'custom': {
                'max_volatility': None,
                'min_sharpe': None,
                'max_single_weight': 1.0,
                'risk_free_rate': 0.05
            }
        }
    
    def calculate_portfolio_metrics(self, weights, returns=None):
        if returns is None:
            returns = self.returns
        
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        risk_free_rate = self.risk_params[self.risk_profile]['risk_free_rate']
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        # Calculate additional metrics
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov() * 252, weights)))
            sortino_ratio = (portfolio_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        # Calculate max drawdown
        cumulative_returns = (1 + (returns * weights).sum(axis=1)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe': sharpe_ratio,
            'sortino': sortino_ratio,
            'max_drawdown': max_drawdown
        }
    
    def optimize_with_risk_constraints(self):
        n_assets = len(self.returns.columns)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        params = self.risk_params[self.risk_profile]
        
        def objective(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            # Maximize Sharpe ratio
            return -metrics['sharpe']
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Add risk constraints based on profile
        if params['max_volatility']:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x: params['max_volatility'] - self.calculate_portfolio_metrics(x)['volatility']
            })
        
        # Weight bounds based on risk profile
        max_weight = params['max_single_weight']
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else initial_weights
    
    def ml_ensemble_optimization(self, model_type='ensemble'):
        available_periods = len(self.returns)
        if available_periods < 60:
            print(f"Insufficient data for ML ({available_periods} days). Using risk-based optimization.")
            return self.optimize_with_risk_constraints()
        
        # Prepare features and targets
        lookback = min(30, available_periods // 3)
        feature_data = []
        target_data = []
        
        for i in range(lookback, len(self.returns) - 1):
            window = self.returns.iloc[i-lookback:i]
            
            features = []
            for col in window.columns:
                col_data = window[col]
                features.extend([
                    col_data.mean(),
                    col_data.std(),
                    col_data.skew() if len(col_data) > 2 else 0,
                    col_data.kurtosis() if len(col_data) > 3 else 0,
                    col_data.min(),
                    col_data.max(),
                    (col_data.iloc[-1] / col_data.iloc[0] - 1) if col_data.iloc[0] != 0 else 0
                ])
            
            feature_data.append(features)
            
            # Next period returns as target
            next_returns = self.returns.iloc[i]
            target_data.append(next_returns.values)
        
        if len(feature_data) < 10:
            return self.optimize_with_risk_constraints()
        
        X = np.array(feature_data)
        y = np.array(target_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        predictions = []
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict for latest window
            latest_window = self.returns.iloc[-lookback:]
            latest_features = self._extract_features(latest_window)
            latest_scaled = scaler.transform(latest_features.reshape(1, -1))
            pred = model.predict(latest_scaled)[0]
            predictions.append(pred)
            
        elif model_type == 'gradient_boost':
            for i in range(y_train.shape[1]):  # One model per asset
                model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X_train_scaled, y_train[:, i])
                
                latest_window = self.returns.iloc[-lookback:]
                latest_features = self._extract_features(latest_window)
                latest_scaled = scaler.transform(latest_features.reshape(1, -1))
                pred = model.predict(latest_scaled)
                predictions.append(pred[0])
            
        elif model_type == 'neural_network':
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            latest_window = self.returns.iloc[-lookback:]
            latest_features = self._extract_features(latest_window)
            latest_scaled = scaler.transform(latest_features.reshape(1, -1))
            pred = model.predict(latest_scaled)[0]
            predictions.append(pred)
            
        else:  # ensemble
            models = [
                RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
                GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
            ]
            
            ensemble_preds = []
            for model in models:
                model.fit(X_train_scaled, y_train)
                latest_window = self.returns.iloc[-lookback:]
                latest_features = self._extract_features(latest_window)
                latest_scaled = scaler.transform(latest_features.reshape(1, -1))
                ensemble_preds.append(model.predict(latest_scaled)[0])
            
            # Average predictions
            predictions.append(np.mean(ensemble_preds, axis=0))
        
        # Convert predictions to weights
        if predictions:
            pred_returns = np.array(predictions).flatten()
            
            # Risk-adjusted weights based on predicted returns
            if self.risk_profile == 'conservative':
                # Favor positive predictions with lower variance
                weights = np.maximum(pred_returns, 0)
                weights = weights / (1 + self.returns.std().values)  # Penalize volatile assets
            elif self.risk_profile == 'aggressive':
                # Favor highest predicted returns
                weights = np.exp(pred_returns * 2)  # Exponential weighting for aggressive
            else:
                # Balanced approach
                weights = np.maximum(pred_returns, 0)
            
            # Normalize
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
            
            # Apply risk constraints
            max_weight = self.risk_params[self.risk_profile]['max_single_weight']
            weights = np.minimum(weights, max_weight)
            weights = weights / weights.sum()
            
            return weights
        
        return self.optimize_with_risk_constraints()
    
    def _extract_features(self, window):
        features = []
        for col in window.columns:
            col_data = window[col]
            features.extend([
                col_data.mean(),
                col_data.std(),
                col_data.skew() if len(col_data) > 2 else 0,
                col_data.kurtosis() if len(col_data) > 3 else 0,
                col_data.min(),
                col_data.max(),
                (col_data.iloc[-1] / col_data.iloc[0] - 1) if col_data.iloc[0] != 0 else 0
            ])
        return np.array(features)
    
    def monte_carlo_simulation(self, n_simulations=1000, n_days=252):
        """Run Monte Carlo simulation for portfolio outcomes"""
        current_weights = self.optimize_with_risk_constraints()
        
        # Historical statistics
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()
        
        portfolio_returns = []
        
        for _ in range(n_simulations):
            # Simulate returns using multivariate normal distribution
            simulated_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, n_days
            )
            
            # Calculate portfolio returns for this simulation
            portfolio_return = np.sum(simulated_returns * current_weights, axis=1)
            cumulative_return = (1 + portfolio_return).cumprod()[-1] - 1
            portfolio_returns.append(cumulative_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        return {
            'expected_return': np.mean(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),  # Value at Risk (95% confidence)
            'var_99': np.percentile(portfolio_returns, 1),  # Value at Risk (99% confidence)
            'best_case': np.percentile(portfolio_returns, 95),
            'worst_case': np.percentile(portfolio_returns, 5),
            'median_return': np.median(portfolio_returns),
            'probability_positive': (portfolio_returns > 0).mean()
        }
    
    def get_rebalancing_recommendations(self, current_weights, target_weights, threshold=0.05):
        """Generate rebalancing recommendations"""
        recommendations = []
        
        for i, symbol in enumerate(self.returns.columns):
            current = current_weights[i]
            target = target_weights[i]
            diff = target - current
            
            if abs(diff) > threshold:
                action = "BUY" if diff > 0 else "SELL"
                recommendations.append({
                    'symbol': symbol,
                    'action': action,
                    'current_weight': f"{current*100:.2f}%",
                    'target_weight': f"{target*100:.2f}%",
                    'change': f"{diff*100:+.2f}%"
                })
        
        return recommendations