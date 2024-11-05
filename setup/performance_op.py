import pandas as pd 
import numpy as np


def compute_aggregated_metrics(results_df: pd.DataFrame) -> pd.DataFrame:   # potential bug 
    """
    Computes weekly, monthly, and yearly returns from the results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame containing individual stock results.

    Returns:
        pd.DataFrame: DataFrame containing aggregated metrics.
    """
    # Ensure 'Date' column is datetime
    results_df['Date'] = pd.to_datetime(results_df['Date'])

    # Calculate daily portfolio return (assuming equal weight per stock)
    daily_returns = results_df.groupby('Date')['Total Return'].mean()

    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod() - 1

    # Resample cumulative returns to get end values for each period
    weekly_cum_returns = cumulative_returns.resample('W-FRI').last()
    monthly_cum_returns = cumulative_returns.resample('M').last()
    yearly_cum_returns = cumulative_returns.resample('Y').last()

    # Compute period returns
    weekly_returns = weekly_cum_returns.pct_change().dropna()
    monthly_returns = monthly_cum_returns.pct_change().dropna()
    yearly_returns = yearly_cum_returns.pct_change().dropna()

    # Prepare DataFrames
    weekly_df = weekly_returns.reset_index()
    weekly_df.columns = ['Date', 'Weekly Return']
    monthly_df = monthly_returns.reset_index()
    monthly_df.columns = ['Date', 'Monthly Return']
    yearly_df = yearly_returns.reset_index()
    yearly_df.columns = ['Date', 'Yearly Return']

    # Merge DataFrames
    aggregated_metrics = pd.merge(weekly_df, monthly_df, on='Date', how='outer')
    aggregated_metrics = pd.merge(aggregated_metrics, yearly_df, on='Date', how='outer')

    # Adjust decimal places to 6
    pd.options.display.float_format = '{:.6f}'.format

    return aggregated_metrics

def save_combined_results(results_df, aggregated_metrics, output_file):     # potential bug
    """
    Saves the individual results and aggregated metrics into one CSV file.

    Args:
        results_df (pd.DataFrame): DataFrame containing individual stock results.
        aggregated_metrics (pd.DataFrame): DataFrame containing aggregated metrics.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, 'w') as f:
        f.write("Individual Stock Results\n")
        results_df.to_csv(f, index=False, float_format='%.6f')
        f.write("\nAggregated Metrics\n")
        aggregated_metrics.to_csv(f, index=False, float_format='%.6f')