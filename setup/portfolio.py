import pandas as pd
import numpy as np


## 計算total 績效

def calculate_portfolio_metrics(results_df: pd.DataFrame, equity_data: dict, total_holding_minutes: int) -> dict:
    """
    Calculates portfolio-level performance metrics across all traded stocks for the day.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing individual stock results.
        equity_data (dict): Dictionary containing equity curves for each stock.
        total_holding_minutes (int): Total holding minutes across all stocks.
        
    Returns:
        dict: Portfolio-level performance metrics.
    """
    if results_df.empty:
        return {}
    
    # Combine all equity curves and calculate portfolio-level returns
    all_equity_curves = pd.DataFrame(equity_data).fillna(method='ffill')
    portfolio_equity = all_equity_curves.sum(axis=1)
    portfolio_returns = portfolio_equity.pct_change().dropna()
    
    # Calculate total trades and total winning trades across all stocks
    total_trades = results_df['Total Trades'].sum()
    
    # Since we may not have individual trade data, estimate winning trades
    # Assuming that if a stock's Total Return > 0, then its trades were winning
    # This is a simplification and may not be accurate if there are multiple trades per stock
    # Alternatively, we can estimate the win rate as the average of individual stock win rates
    win_rate = results_df['Win Rate'].mean()
    
    # Maximum Drawdown calculation for the portfolio
    portfolio_cummax = portfolio_equity.cummax()
    drawdowns = (portfolio_cummax - portfolio_equity) / portfolio_cummax
    max_drawdown = drawdowns.max()
    
    # Total Return for the portfolio
    total_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
    
    # Calmar Ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else 0
    
    # PnL ratio calculation
    positive_returns = portfolio_returns[portfolio_returns > 0].sum()
    negative_returns = -portfolio_returns[portfolio_returns < 0].sum()
    pnl_ratio = positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    # Volatility and risk metrics
    daily_volatility = portfolio_returns.std()
    
    # Sharpe ratio (assuming risk-free rate = 0 for intraday)
    daily_sharpe = total_return / daily_volatility if daily_volatility != 0 else 0
    
    # Sortino ratio (using only negative returns for denominator)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std()
    daily_sortino = total_return / downside_std if downside_std != 0 else 0
    
    # Calculate average holding time
    avg_holding_minutes = total_holding_minutes / total_trades if total_trades > 0 else 0
    
    return {
        'Date': results_df['Date'].iloc[0],
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Maximum Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'PnL Ratio': pnl_ratio,
        'Daily Sharpe': daily_sharpe,
        'Daily Sortino': daily_sortino,
        'Daily Volatility': daily_volatility,
        'Total Return': total_return,
        'Average Holding Minutes': avg_holding_minutes,
        'Total Holding Minutes': total_holding_minutes
    }
