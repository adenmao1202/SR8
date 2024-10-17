import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pyarrow.feather as feather
import multiprocessing as mp
import argparse
import yaml
from datetime import datetime, time
import logging
from functools import partial
from numba import jit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a global DataFrame to store all results
all_results = pd.DataFrame()

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def read_stock_data(file_path: str, config: dict) -> pd.DataFrame:
    """
    Read stock data from a CSV file or a Feather file if it exists.

    Args:
        file_path (str): Path to the CSV file.
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: Processed stock data.
    """
    feather_file = Path(file_path).with_suffix('.feather')

    try:
        if feather_file.exists():
            logger.info(f"Reading Feather file: {feather_file}")
            df = feather.read_feather(feather_file)
        else:
            logger.info(f"Feather file not found. Reading CSV and converting to Feather: {file_path}")
            df = pd.read_csv(file_path)
            
            required_columns = config['required_columns']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file {file_path} must have columns: {', '.join(required_columns)}")

            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            df.sort_index(inplace=True)

            if df.index.inferred_freq != '1min':
                df = df.resample('1min').last().ffill()

            feather.write_feather(df.reset_index(), feather_file)

        df.set_index('ts', inplace=True)
        return df

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


def construct_data(data: pd.DataFrame, start_time: datetime, end_time: datetime, config: dict) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Construct numpy arrays from the input data, including VWAP, moving average, and buy/sell pressure.

    Args:
        data (pd.DataFrame): Input stock data.
        start_time (datetime): Start time for the analysis.
        end_time (datetime): End time for the analysis.
        config (dict): Configuration parameters.

    Returns:
        Tuple[int, Dict[str, np.ndarray]]: Length of data, and numpy arrays.
    """
    # Filter the data based on time range
    data = data[(data.index >= start_time) & (data.index < end_time)].reset_index()
    idx_len = len(data)

    # Convert necessary columns to NumPy arrays
    converting_columns = config['price_columns']
    price_data_np = {col: data[col].to_numpy(dtype=np.float64) for col in converting_columns}
    
    # Convert the timestamp to minutes since start of the day for easier time manipulation
    price_data_np['open_time'] = (data['ts'].dt.hour * 60 + data['ts'].dt.minute).to_numpy(dtype=np.int32)
    
    # Calculate moving averages and VWAP using NumPy
    ma_period = config['ma_period']
    price_data_np['MA'] = np.convolve(price_data_np['Close'], np.ones(ma_period), 'valid') / ma_period
    
    # Calculate VWAP (Volume Weighted Average Price)
    price_data_np['VWAP'] = np.cumsum(price_data_np['Volume'] * price_data_np['Close']) / np.cumsum(price_data_np['Volume'])
    
    # Calculate Buy and Sell Pressure based on VWAP
    price_data_np['BuyPressure'] = price_data_np['Volume'] * (price_data_np['Close'] > price_data_np['VWAP']).astype(np.float64)
    price_data_np['SellPressure'] = price_data_np['Volume'] * (price_data_np['Close'] < price_data_np['VWAP']).astype(np.float64)
    
    return idx_len, price_data_np


@jit(nopython=True)
def execute_short_strategy_optimized(open_np, high_np, low_np, close_np, ma_np, buy_pressure_np, sell_pressure_np, open_time_np,
                                     initial_cap, fee_rate, stop_loss_pct, take_profit_pct, threshold, entry_k, 
                                     entry_time, exit_times):
    """
    Optimized short strategy using NumPy arrays and compiled with Numba for faster execution.
    """
    idx_len = len(open_np)
    cap = np.full(idx_len, np.nan, dtype=np.float64)
    cap[0] = initial_cap
    equity = np.full(idx_len, np.nan, dtype=np.float64)
    equity[0] = initial_cap
    short_position = np.zeros(idx_len, dtype=np.float64)
    short_entry_price = 0.0
    trade_count = 0

    # Calculate masks for entry and exit times
    first_enter_mask = (open_time_np == entry_time)
    first_kbars_mask = (open_time_np == 540)  # 540 corresponds to 9:00 (9*60)
    exit_times_mask = np.isin(open_time_np, exit_times)

    allow_entry_today = False

    for i in range(1, idx_len - 1):
        # Check if entry is allowed based on the first K-bar
        if first_kbars_mask[i]:
            if open_np[i] / close_np[i-1] > 1 + threshold:
                allow_entry_today = True
            else:
                allow_entry_today = False

        # Entry logic at the specified time and if entry is allowed today
        if first_enter_mask[i] and allow_entry_today:
            if cap[i] > 0 and short_position[i-1] == 0:
                short_entry_price = open_np[i]
                short_position[i] = cap[i] / (short_entry_price * (1 + fee_rate))
                cap[i] = 0
                trade_count += 1

        # Exit logic (based on stop loss, take profit, or exit times)
        if short_position[i] > 0:
            # Stop loss condition
            if (close_np[i] >= short_entry_price * (1 + stop_loss_pct)) or \
               (i > 0 and sell_pressure_np[i-1] > buy_pressure_np[i-1] and sell_pressure_np[i] <= buy_pressure_np[i]) or \
               (low_np[i] <= short_entry_price * (1 - take_profit_pct)):
                exit_price = open_np[i+1]
                cap[i+1] = short_position[i] * (2 * short_entry_price - exit_price * (1 + fee_rate))
                short_position[i+1] = 0
                continue

            # Exit at the specified times (partial exit)
            if exit_times_mask[i]:
                short_exit_price = open_np[i]
                cap[i] += (short_position[i] / 3) * (2 * short_entry_price - short_exit_price * (1 + fee_rate))
                short_position[i] *= (2/3)
        
        # Update capital and equity
        equity[i] = cap[i]
        if short_position[i] > 0:
            equity[i] += short_position[i] * (2 * short_entry_price - close_np[i])

        # Carry over capital and position to the next time step
        cap[i + 1] = cap[i]
        short_position[i + 1] = short_position[i]

    # Final update at the last time step
    equity[idx_len-1] = cap[idx_len-1] + short_position[idx_len-1] * (2 * short_entry_price - close_np[idx_len-1])
    
    return equity, trade_count


def execute_short_strategy(idx_len: int, price_data_np: Dict[str, np.ndarray], config: dict) -> Tuple[np.ndarray, int]:
    """
    Wrapper to execute the optimized short strategy on the given price data.
    """
    return execute_short_strategy_optimized(
        price_data_np['Open'], price_data_np['High'], price_data_np['Low'], price_data_np['Close'],
        price_data_np['MA'], price_data_np['BuyPressure'], price_data_np['SellPressure'], price_data_np['open_time'],
        config['initial_cap'], config['fee_rate'], config['stop_loss_pct'], config['take_profit_pct'],
        config['threshold_range'], config['entry_range'], config['entry_time'], config['exit_times']
    )


def calculate_performance_metrics(equity: np.ndarray, price_data_np: Dict[str, np.ndarray], trade_count: int, config: dict) -> Dict[str, float]:
    """
    Calculate performance metrics for the executed strategy.
    """
    returns = np.diff(equity) / equity[:-1]
    buy_hold_returns = np.diff(price_data_np['Close']) / price_data_np['Close'][:-1]
    
    total_return = (equity[-1] / equity[0]) - 1
    annualized_return = (1 + total_return) ** (config['trading_days_per_year'] / len(equity)) - 1
    sharpe_ratio = np.sqrt(config['trading_days_per_year']) * (np.mean(returns) - config['risk_free_rate']) / np.std(returns)
    
    drawdowns = 1 - equity / np.maximum.accumulate(equity)
    max_drawdown = np.max(drawdowns)
    
    win_rate = np.sum(returns > 0) / len(returns)
    
    return {
        'Total Trades': trade_count,
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Strategy vs Buy&Hold': (1 + total_return) / (1 + (price_data_np['Close'][-1] / price_data_np['Close'][0] - 1)) - 1
    }


def run_backtest(file_path: str, config: dict) -> Dict[str, float]:
    """
    Run a backtest for a single stock.
    """
    try:
        data = read_stock_data(file_path, config)
        idx_len, price_data_np = construct_data(data, config['start_time'], config['end_time'], config)
        equity, trade_count = execute_short_strategy(idx_len, price_data_np, config)
        metrics = calculate_performance_metrics(equity, price_data_np, trade_count, config)
        metrics['File'] = Path(file_path).name
        return metrics
    except Exception as e:
        logger.error(f"Error in backtesting {file_path}: {str(e)}")
        return None


def process_batch(files: List[Path], config: dict) -> pd.DataFrame:
    """
    Process a batch of stock files.
    """
    with mp.Pool(processes=config['num_processes']) as pool:
        results = pool.map(partial(run_backtest, config=config), files)
    
    return pd.DataFrame([r for r in results if r is not None])


def main(config_path: str):
    """
    Main function to run the backtesting system.
    """
    global all_results  # To access and update the global DataFrame
    config = load_config(config_path)
    data_folder = Path('/Users/mouyasushi/k_data/永豐')  # Your specified path
    all_files = list(data_folder.glob('*.csv'))  

    for i in range(0, len(all_files), config['batch_size']):
        batch_files = all_files[i:i+config['batch_size']]
        batch_results_df = process_batch(batch_files, config)
        all_results = pd.concat([all_results, batch_results_df], ignore_index=True)
        
        # Show the results for the current batch
        for index, row in batch_results_df.iterrows():
            logger.info(f"\nFile: {row['File']}")
            logger.info(f"Total Trades: {row['Total Trades']}")
            logger.info(f"Total Return: {row['Total Return']}")
            logger.info(f"Annualized Return: {row['Annualized Return']}")
            logger.info(f"Sharpe Ratio: {row['Sharpe Ratio']}")
            logger.info(f"Max Drawdown: {row['Max Drawdown']}")
            logger.info(f"Win Rate: {row['Win Rate']}")
            logger.info(f"Strategy vs Buy&Hold: {row['Strategy vs Buy&Hold']}")
    
    # Check if 'Total Return' column exists before sorting
    if 'Total Return' in all_results.columns:
        all_results = all_results.sort_values('Total Return', ascending=False)
    else:
        logger.warning("'Total Return' column not found, skipping sorting.")

    # Only calculate mean for numeric columns
    numeric_columns = all_results.select_dtypes(include=[np.number]).columns
    average_metrics = all_results[numeric_columns].mean()
    average_metrics['File'] = 'Average'
    all_results = pd.concat([all_results, pd.DataFrame([average_metrics])], ignore_index=True)

    logger.info("\nBacktesting Results:")
    logger.info(all_results.to_string(index=False))

    output_file = data_folder / 'backtest_results_final.csv'
    all_results.to_csv(output_file, index=False)
    logger.info(f"\nFinal results saved to {output_file}")
    
    return all_results  # Return the final DataFrame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock backtesting system")
    parser.add_argument("config", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)
