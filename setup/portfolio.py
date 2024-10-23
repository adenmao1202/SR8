import pandas as pd
import numpy as np


def calculate_portfolio_metrics(results_df: pd.DataFrame, equity_data: dict) -> dict:
    """
    Calculates portfolio-level performance metrics across all traded stocks for the day.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing individual stock results
        equity_data (dict): Dictionary containing equity curves for each stock
        
    Returns:
        dict: Portfolio-level performance metrics
    """
    if results_df.empty:
        return {}
    
    # Combine all equity curves and calculate portfolio-level returns
    all_equity_curves = pd.DataFrame(equity_data).fillna(method='ffill')
    portfolio_equity = all_equity_curves.sum(axis=1)
    portfolio_returns = portfolio_equity.pct_change().dropna()
    
    # Calculate portfolio-level metrics
    total_trades = results_df['Total Trades'].sum()
    winning_trades = results_df[results_df['Total Return'] > 0].shape[0]
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Maximum Drawdown calculation
    portfolio_cummax = portfolio_equity.cummax()
    drawdowns = (portfolio_cummax - portfolio_equity) / portfolio_cummax
    max_drawdown = drawdowns.max()
    
    # Daily return metrics
    daily_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
    
    # Calmer ratio
    calmer_ratio = daily_return / max_drawdown if max_drawdown != 0 else 0
    
    # PnL ratio calculation
    positive_returns = portfolio_returns[portfolio_returns > 0].sum()
    negative_returns = abs(portfolio_returns[portfolio_returns < 0].sum())
    pnl_ratio = positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    # Volatility and risk metrics
    daily_volatility = portfolio_returns.std()
    
    # Sharpe ratio (assuming risk-free rate = 0 for intraday)
    daily_sharpe = daily_return / daily_volatility if daily_volatility != 0 else 0
    
    # Sortino ratio (using only negative returns for denominator)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std()
    daily_sortino = daily_return / downside_std if downside_std != 0 else 0
    
    # Average holding time calculation
    total_holding_minutes = sum(len(equity_curve) for equity_curve in equity_data.values())
    avg_holding_minutes = total_holding_minutes / total_trades if total_trades > 0 else 0
    
    return {
        'Date': results_df['Date'].iloc[0],
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Maximum Drawdown': max_drawdown,
        'Calmer Ratio': calmer_ratio,
        'PnL Ratio': pnl_ratio,
        'Daily Sharpe': daily_sharpe,
        'Daily Sortino': daily_sortino,
        'Daily Volatility': daily_volatility,
        'Total Return': daily_return,
        'Average Holding Minutes': avg_holding_minutes
    }