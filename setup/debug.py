import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from numba import jit
import yaml
import argparse
import ast
import os

# Constants
CSV_PATH = "/Users/mouyasushi/k_data/永豐"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper functions
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def read_stock_data(stock_code: str, date: datetime) -> pd.DataFrame:
    file_path = Path(CSV_PATH) / f"{stock_code}.csv"
    
    if not file_path.exists():
        logger.error(f"File for stock {stock_code} not found at {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df[df.index.date == date.date()]

def construct_data(data: pd.DataFrame, ma_period: int) -> dict:
    if data.empty:
        logger.warning(f"No data available for the specified date")
        return {}

    data['MA'] = data['Close'].rolling(window=ma_period).mean()
    data['VWAP'] = np.cumsum(data['Volume'] * data['Close']) / np.cumsum(data['Volume'])
    
    data_np = {col: data[col].to_numpy() for col in ['Open', 'Close', 'High', 'Low', 'Volume', 'MA', 'VWAP']}
    data_np['open_time'] = (data.index.hour * 60 + data.index.minute).to_numpy(dtype=np.int32)
    
    return data_np

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

entry_time_str = config['entry_time']
entry_time_min = int(entry_time_str.split(':')[0]) * 60 + int(entry_time_str.split(':')[1])

# Strategy execution
@jit(nopython=True)
def execute_short_strategy(open_np, close_np, vwap_np, open_time_np, initial_cap, fee_rate, stop_loss_pct, threshold):
    idx_len = len(open_np)
    equity = np.full(idx_len, initial_cap)
    short_position = np.zeros(idx_len)
    entry_price = 0.0
    trades = 0
    cap = initial_cap
    entry_executed = False

    if idx_len < 6:
        return equity, trades

    previous_day_last_close = close_np[0]
    first_5_open_avg = np.mean(open_np[:5])
    
    if first_5_open_avg / previous_day_last_close <= (1 + threshold):
        return equity, trades

    for i in range(1, idx_len):
        if open_time_np[i] == entry_time_min and not entry_executed:
            entry_price = open_np[i]
            short_position[i] = cap / (entry_price * (1 + fee_rate))
            cap = 0
            trades += 1
            entry_executed = True
            continue

        if short_position[i] > 0:
            if close_np[i] > vwap_np[i]:
                cap = short_position[i] * (2 * entry_price - open_np[i] * (1 + fee_rate))
                short_position[i] = 0
                continue
            
            if close_np[i] <= entry_price * (1 + stop_loss_pct):
                cap = short_position[i] * (2 * entry_price - open_np[i] * (1 + fee_rate))
                short_position[i] = 0
                continue

            if i == idx_len - 1:
                cap += short_position[i] * (2 * entry_price - open_np[i] * (1 + fee_rate))
                short_position[i] = 0

        equity[i] = cap + short_position[i] * (2 * entry_price - close_np[i])

    return equity, trades

# Performance calculation
def calculate_daily_performance(equity: np.ndarray, price_data_np: dict, trade_count: int) -> dict:
    returns = np.diff(equity) / equity[:-1]
    total_return = (equity[-1] / equity[0]) - 1
    
    drawdowns = 1 - equity / np.maximum.accumulate(equity)
    max_drawdown = np.max(drawdowns)
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    
    daily_volatility = np.std(returns) if len(returns) > 0 else 0
    
    return {
        'Total Trades': trade_count,
        'Total Return': total_return,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Daily Volatility': daily_volatility
    }

# Backtesting functions
def backtest_single_stock(stock_code: str, date: datetime, config: dict) -> dict:
    logger.info(f"Running backtest for stock: {stock_code} on date: {date}")

    stock_data = read_stock_data(stock_code, date)

    if stock_data.empty:
        logger.warning(f"No data for stock: {stock_code} on date: {date}")
        return {}

    data_np = construct_data(stock_data, config['ma_period'])

    if not data_np:
        logger.warning(f"No data constructed for stock: {stock_code} on date: {date}")
        return {}

    equity, trades = execute_short_strategy(
        data_np['Open'], data_np['Close'], data_np['VWAP'], data_np['open_time'],
        config['initial_cap'], config['fee_rate'], config['stop_loss_pct'],
        config['threshold']
    )

    performance = calculate_daily_performance(equity, data_np, trades)
    performance['Stock'] = stock_code
    performance['Date'] = date.date()
    return performance

def get_stocks_for_date(stock_list_file: str, date: datetime) -> list:
    logger.info(f"Reading stock list from file: {stock_list_file}")
    logger.info(f"Searching for date: {date.date()}")

    if not os.path.exists(stock_list_file):
        logger.error(f"Stock list file {stock_list_file} does not exist.")
        return []

    try:
        # Force 'StockCode' to be read as a string
        df = pd.read_csv(stock_list_file, dtype={'StockCode': str}, encoding='utf-8')
        logger.info(f"CSV file read successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        if 'Date' not in df.columns or 'StockCode' not in df.columns:
            logger.error("Required columns 'Date' and 'StockCode' not found in the CSV file")
            return []

        df['Date'] = pd.to_datetime(df['Date'])
        stock_codes = df[df['Date'].dt.date == date.date()]['StockCode'].values
        
        if len(stock_codes) > 0:
            stock_code_str = stock_codes[0]
            
            # Handle stock codes safely as strings
            if isinstance(stock_code_str, str):
                try:
                    stock_codes = ast.literal_eval(stock_code_str)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing stock codes for date {date.date()}: {str(e)}")
                    return []
            else:
                stock_codes = list(stock_code_str)
            
            logger.info(f"Stocks for date {date.date()}: {stock_codes}")
            return stock_codes
        else:
            logger.warning(f"No stocks found for date: {date.date()}")
            return []
    except Exception as e:
        logger.error(f"Error reading or processing the stock list file: {str(e)}")
        return []


def backtest_multiple_stocks(stocks: list, date: datetime, config: dict) -> list:
    results = []
    for stock_code in stocks:
        result = backtest_single_stock(stock_code, date, config)
        if result:
            results.append(result)
    return results

# Main function
def main(config_path: str, date_str: str, stock_list_file: str):
    config = load_config(config_path)
    
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Date string provided: {date_str}")
    logger.info(f"Stock list file: {stock_list_file}")
    
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        logger.info(f"Parsed date: {date}")
    except ValueError as e:
        logger.error(f"Error parsing date: {str(e)}")
        return
    
    stocks = get_stocks_for_date(stock_list_file, date)
    
    if stocks:
        results = backtest_multiple_stocks(stocks, date, config)
        
        if results:
            logger.info("\nBacktesting Results:")
            for result in results:
                for key, value in result.items():
                    logger.info(f"{key}: {value}")
                logger.info("---")

            output_file = f'backtest_results_{date.date()}.csv'
            pd.DataFrame(results).to_csv(output_file, index=False)
            logger.info(f"\nResults saved to {output_file}")
        else:
            logger.warning("No valid results from backtesting.")
    else:
        logger.warning("No stocks found for the specified date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stock backtesting")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("date", help="Date for backtesting (YYYY-MM-DD)")
    parser.add_argument("stock_list_file", help="Path to file containing stock codes and dates")
    args = parser.parse_args()

    main(args.config, args.date, args.stock_list_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stock backtesting")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("date", help="Date for backtesting (YYYY-MM-DD)")
    parser.add_argument("stock_list_file", help="Path to file containing stock codes and dates")
    args = parser.parse_args()

    main(args.config, args.date, args.stock_list_file)