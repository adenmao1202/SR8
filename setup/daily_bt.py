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
from multiprocessing import Pool

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict: # -- > okay 
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration settings as a dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Validate required fields
        required_fields = [
            'csv_path', 'output_directory', 'ma_period', 'initial_cap',
            'fee_rate', 'stop_loss_pct', 'threshold', 'entry_time'
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {missing_fields}")
            
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}

def read_stock_data(stock_code: str, date: datetime, csv_path: str) -> pd.DataFrame:
    """
    Reads stock data for a given stock code and date.

    Args:
        stock_code (str): Stock code to read data for.
        date (datetime): Date for which to read data.
        csv_path (str): Path to the directory containing stock CSV files.

    Returns:
        pd.DataFrame: DataFrame containing stock data for the specified date.
    """
    file_path = Path(csv_path) / f"{stock_code}.csv"

    if not file_path.exists():
        logger.error(f"File for stock {stock_code} not found at {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        if 'ts' not in df.columns:
            logger.error(f"'ts' column not found in {file_path}")
            return pd.DataFrame()
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        df.sort_index(inplace=True)
        df = df[df.index.date == date.date()]
        if df.empty:
            logger.warning(f"No data for stock {stock_code} on date {date.date()}")
        return df
    except Exception as e:
        logger.error(f"Error reading stock data for {stock_code}: {e}")
        return pd.DataFrame()

def construct_data(data: pd.DataFrame, ma_period: int) -> dict:
    """
    Constructs necessary data arrays for backtesting, standardizing the data
    to fixed trading hours and filling missing times with zeros.

    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        ma_period (int): Period for moving average calculation.

    Returns:
        dict: Dictionary containing NumPy arrays of required data.
    """
    if data.empty:
        logger.warning(f"No data available for the specified date")
        return {}

    # Set the trading hours from 9:01 a.m. to 1:30 p.m.
    trading_start = datetime.combine(data.index[0].date(), datetime.strptime('09:01', '%H:%M').time())
    trading_end = datetime.combine(data.index[0].date(), datetime.strptime('13:30', '%H:%M').time())
    time_index = pd.date_range(start=trading_start, end=trading_end, freq='1T')

    # Reindex the DataFrame to include all minutes in the trading hours
    data = data.reindex(time_index)

    # Fill missing values with zeros
    data.fillna(0, inplace=True)

    # Calculate Moving Average (MA) and Volume Weighted Average Price (VWAP)
    data['MA'] = data['Close'].rolling(window=ma_period, min_periods=1).mean()
    data['VWAP'] = (data['Volume'] * data['Close']).cumsum() / data['Volume'].cumsum()

    # Since we filled missing values with zeros, ensure MA and VWAP are calculated correctly
    # Replace zero volumes with NaN to avoid division by zero in VWAP calculation
    data['Volume'].replace(0, np.nan, inplace=True)
    data['VWAP'] = (data['Volume'].fillna(0) * data['Close']).cumsum() / data['Volume'].cumsum()
    data['Volume'].fillna(0, inplace=True)

    # Handle potential NaN values in MA and VWAP
    data['MA'].fillna(method='ffill', inplace=True)
    data['VWAP'].fillna(method='ffill', inplace=True)

    # Convert index to minutes since midnight for open_time_np
    data_np = {col: data[col].to_numpy() for col in ['Open', 'Close', 'High', 'Low', 'Volume', 'MA', 'VWAP']}
    data_np['open_time'] = (data.index.hour * 60 + data.index.minute).to_numpy(dtype=np.int32)

    return data_np


def execute_short_strategy(open_np, close_np, vwap_np, open_time_np, initial_cap, fee_rate, 
                           stop_loss_pct, threshold, entry_time_min, previous_day_last_close):
    """
    Executes the short-selling strategy with proper accounting.
    """
    idx_len = len(open_np)  # open_np will have diff value everyday 
    equity = np.full(idx_len, initial_cap)
    position_size = 0.0
    trades = 0
    cap = initial_cap  # Cash balance including proceeds from short sale
    has_traded_today = False
    initial_position_size = 0.0
    entry_price = None
    partial_exit_times = [660, 780, 810]
    partial_exit_index = 0

    if idx_len < 5:
        return equity, trades

    first_5_open_avg = np.mean(open_np[:5])  # 看前五根k bar 的avg 決定是否進場 

    if first_5_open_avg / previous_day_last_close <= threshold:
        return equity, trades

    for i in range(idx_len):
        current_time = open_time_np[i]

        if current_time >= entry_time_min and i >= 5 and not has_traded_today:
            # Enter short position
            entry_price = open_np[i]
            position_size = initial_cap / entry_price
            transaction_fee = position_size * entry_price * fee_rate
            short_proceeds = position_size * entry_price
            cap = initial_cap + short_proceeds - transaction_fee
            initial_position_size = position_size
            trades += 1
            has_traded_today = True

        if has_traded_today and position_size > 0:
            current_price = close_np[i]
            # Correct calculation of equity for a short position
            equity[i] = cap - (position_size * current_price)

            # Exit conditions
            if current_price > vwap_np[i]:
                # Exit entire position
                exit_price = current_price
                transaction_fee = position_size * exit_price * fee_rate
                cap -= (position_size * exit_price) + transaction_fee
                position_size = 0
                equity[i] = cap
                continue

            if current_price >= entry_price * (1 + stop_loss_pct):
                # Exit entire position due to stop loss
                exit_price = current_price
                transaction_fee = position_size * exit_price * fee_rate
                cap -= (position_size * exit_price) + transaction_fee
                position_size = 0
                equity[i] = cap
                continue

            # Partial exits
            while (partial_exit_index < len(partial_exit_times) and
                   current_time >= partial_exit_times[partial_exit_index] and
                   position_size > 0):
                portion_to_close = min(initial_position_size / 3, position_size)
                exit_price = current_price
                transaction_fee = portion_to_close * exit_price * fee_rate
                cap -= (portion_to_close * exit_price) + transaction_fee
                position_size -= portion_to_close
                equity[i] = cap - (position_size * current_price)
                partial_exit_index += 1

                if position_size <= 0:
                    # Position fully closed
                    break

            if i == idx_len - 1 and position_size > 0:
                # Exit remaining position at end of day
                exit_price = current_price
                transaction_fee = position_size * exit_price * fee_rate
                cap -= (position_size * exit_price) + transaction_fee
                position_size = 0
                equity[i] = cap
        else:
            # No position; equity remains the same
            equity[i] = cap

    return equity, trades




def calculate_daily_performance(equity: np.ndarray, trade_count: int) -> dict:
    """
    Calculates daily performance metrics.
    """
    returns = np.diff(equity) / equity[:-1]
    total_return = (equity[-1] / equity[0]) - 1 if equity[0] != 0 else 0

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


def get_stocks_for_date(stock_list_file: str, date: datetime) -> list:
    """
    Retrieves the list of stocks for the specified date.
    """
    logger.info(f"Reading stock list from Feather file: {stock_list_file}")

    try:
        df = pd.read_feather(stock_list_file)
        if 'index' not in df.columns or 'stock_list' not in df.columns:
            logger.error("Feather file must contain 'index' and 'stock_list' columns")
            return []
        df.rename(columns={'index': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        stock_codes_series = df[df['Date'].dt.date == date.date()]['stock_list']
        if not stock_codes_series.empty:
            stock_code_str = stock_codes_series.iloc[0]
            if isinstance(stock_code_str, str):
                try:
                    stock_codes = ast.literal_eval(stock_code_str)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing stock codes for date {date.date()}: {e}")
                    return []
            elif isinstance(stock_code_str, list):
                stock_codes = stock_code_str
            else:
                logger.error(f"Unexpected data type for stock codes: {type(stock_code_str)}")
                return []
            logger.info(f"Found {len(stock_codes)} stocks for date {date.date()}")
            return stock_codes
        else:
            logger.warning(f"No stocks found for date: {date.date()}")
            return []
    except Exception as e:
        logger.error(f"Error reading stock list: {e}")
        return []

def backtest_single_stock(stock_code: str, date: datetime, config: dict) -> dict:
    """
    Runs backtest for a single stock.
    """
    logger.info(f"Running backtest for stock: {stock_code} on date: {date.date()}")

    csv_path = config.get('csv_path', '')
    stock_data = read_stock_data(stock_code, date, csv_path)

    if stock_data.empty:
        logger.warning(f"No data for stock: {stock_code} on date: {date.date()}")
        return {}

    data_np = construct_data(stock_data, config['ma_period'])

    if not data_np:
        logger.warning(f"No data constructed for stock: {stock_code} on date: {date.date()}")
        return {}

    # Get previous day's close price
    previous_date = date - pd.Timedelta(days=1)
    previous_data = read_stock_data(stock_code, previous_date, csv_path)
    if previous_data.empty:
        logger.warning(f"No data for stock: {stock_code} on previous date: {previous_date.date()}")
        return {}
    previous_day_last_close = previous_data['Close'].iloc[-1]

    entry_time_str = config['entry_time']
    entry_time_min = int(entry_time_str.split(':')[0]) * 60 + int(entry_time_str.split(':')[1])

    equity, trades = execute_short_strategy(
        data_np['Open'], data_np['Close'], data_np['VWAP'], data_np['open_time'],
        config['initial_cap'], config['fee_rate'], config['stop_loss_pct'],
        config['threshold'], entry_time_min, previous_day_last_close
    )

    performance = calculate_daily_performance(equity, trades)
    performance['Stock'] = stock_code
    performance['Date'] = date.date()
    return performance

def backtest_multiple_stocks(stocks: list, date: datetime, config: dict) -> list:
    """
    Runs backtest for multiple stocks in parallel.
    """
    def backtest_wrapper(stock_code):
        return backtest_single_stock(stock_code, date, config)

    with Pool(processes=config.get('num_processes', 4)) as pool:
        results = pool.map(backtest_wrapper, stocks)
    
    return [result for result in results if result]

def main(config_path: str, date_str: str, stock_list_file: str):
    """
    Main function to execute backtesting.
    """
    try:
        config = load_config(config_path)
        if not config:
            logger.error("Failed to load configuration")
            return

        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("Date must be in YYYY-MM-DD format")
            return

        stocks = get_stocks_for_date(stock_list_file, date)
        if not stocks:
            logger.warning("No stocks found for the specified date")
            return

        results = backtest_multiple_stocks(stocks, date, config)

        if results:
            output_directory = config.get('output_directory', '.')
            os.makedirs(output_directory, exist_ok=True)
            
            output_file = Path(output_directory) / f'backtest_results_{date.date()}.csv'
            pd.DataFrame(results).to_csv(output_file, index=False)
            
            logger.info("\nBacktesting Results:")
            results_df = pd.DataFrame(results)
            logger.info(f"Total stocks tested: {len(results_df)}")
            logger.info(f"Average return: {results_df['Total Return'].mean():.2%}")
            logger.info(f"Average max drawdown: {results_df['Max Drawdown'].mean():.2%}")
            logger.info(f"Results saved to {output_file}")
        else:
            logger.warning("No valid results from backtesting")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stock backtesting")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("date", help="Date for backtesting (YYYY-MM-DD)")
    parser.add_argument("stock_list_file", help="Path to file containing stock codes and dates")
    args = parser.parse_args()

    try:
        main(args.config, args.date, args.stock_list_file)
    except Exception as e:
        logger.error(f"Program terminated with error: {e}")
        exit(1)