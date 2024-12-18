import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import yaml
import argparse
import ast
import os
from multiprocessing import Pool
import warnings
import numba
warnings.filterwarnings("ignore")

""" 
Final version for now 
"""




# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
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
            'fee_rate', 'stop_loss_pct', 'threshold', 'entry_time', 'num_processes'
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


######################################################################################

@numba.njit   
def execute_short_strategy(open_np, close_np, vwap_np, open_time_np, initial_cap, fee_rate, 
                           stop_loss_pct, threshold, entry_time_min, previous_day_last_close):
    """
    Executes the short-selling strategy with proper accounting.
    """
    idx_len = len(open_np)
    equity = np.full(idx_len, initial_cap)
    position_size = 0.0
    trades = 0   # dummy variable
    cap = initial_cap
    has_traded_today = False
    initial_position_size = 0.0
    entry_price = 0.0  # Initialize to 0.0 instead of None
    partial_exit_times = np.array([660, 780, 810], dtype=np.int32)
    partial_exit_index = 0

    # Add holding time tracking
    holding_minutes = 0
    entry_time = -1  # Initialize to -1 instead of None

    
    # first 5 bars don't enter 
    if idx_len < 6:
        return equity, trades, holding_minutes

    # Compute first_5_open_avg, skip zeros
    valid_opens = open_np[:5][open_np[:5] != 0.0]
    if len(valid_opens) == 0:
        return equity, trades, holding_minutes
    first_5_open_avg = np.mean(valid_opens)

    if first_5_open_avg / previous_day_last_close <= threshold:
        return equity, trades, holding_minutes

    for i in range(idx_len):
        current_time = open_time_np[i]

        if open_np[i] == 0.0 or close_np[i] == 0.0:
            equity[i] = cap
            continue

        # Enter position
        if current_time >= entry_time_min and i >= 5 and not has_traded_today:
            entry_price = open_np[i]
            position_size = initial_cap / entry_price
            initial_position_size = position_size  # Set initial position size here
            transaction_fee = position_size * entry_price * fee_rate
            cap -= transaction_fee
            trades += 1
            has_traded_today = True
            entry_time = current_time  # Record entry time

        if has_traded_today and position_size > 0.0:
            current_price = close_np[i]
            equity[i] = cap + (position_size * (entry_price - current_price))

            # Exit conditions
            exit_position = False

            # Check all exit conditions
            # if (current_price > vwap_np[i] or  # VWAP breach
            if (current_price >= entry_price * (1.0 + stop_loss_pct) or  # Stop loss
                i == idx_len - 1):  # End of day
                exit_position = True

            # Handle partial exits
            while (partial_exit_index < len(partial_exit_times) and
                   current_time >= partial_exit_times[partial_exit_index] and
                   position_size > 0.0):
                portion_to_close = min(initial_position_size / 3.0, position_size)
                # If this is the last portion, treat it as a full exit
                if position_size - portion_to_close <= 0.0:
                    exit_position = True
                    break
                # Otherwise, handle partial exit
                exit_price = current_price
                transaction_fee = portion_to_close * exit_price * fee_rate
                cap += portion_to_close * (entry_price - exit_price) - transaction_fee
                position_size -= portion_to_close
                equity[i] = cap + (position_size * (entry_price - current_price))
                partial_exit_index += 1

            # Handle full position exit
            if exit_position:
                exit_price = current_price
                transaction_fee = position_size * exit_price * fee_rate
                cap += position_size * (entry_price - exit_price) - transaction_fee
                # Calculate holding time when position is closed
                holding_minutes += current_time - entry_time
                position_size = 0.0
                equity[i] = cap
        else:
            equity[i] = cap

    return equity, trades, holding_minutes


def calculate_daily_performance(equity: np.ndarray, trade_count: int) -> dict:
    """
    Calculates daily performance metrics.
    """
    # For short strategy, positive returns happen when prices fall
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


# Assuming logger is already configured
logger = logging.getLogger(__name__)

def get_stocks_for_date(stock_list_file: str, date: datetime) -> list:
    """
    Retrieves the list of stocks for the specified date.
    """
    logger.info(f"Reading stock list from Parquet file: {stock_list_file}")

    try:
        # Read the Parquet file into a DataFrame
        df = pd.read_parquet(stock_list_file)

        # Ensure 'stock_list' column is present
        if 'stock_list' not in df.columns:
            logger.error("Parquet file must contain 'stock_list' column")
            return []

        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be a DatetimeIndex")
            return []

        # Convert index to dates for comparison
        df.index = df.index.normalize()

        # Filter the DataFrame for the specified date
        target_date = pd.Timestamp(date.date())
        stock_codes_series = df.loc[df.index == target_date, 'stock_list']

        if not stock_codes_series.empty:
            # Extract the first (and should be only) entry for the date
            stock_code_entry = stock_codes_series.iloc[0]

            # Debugging: Log the type and content of stock_code_entry
            logger.debug(f"stock_code_entry type: {type(stock_code_entry)}")
            logger.debug(f"stock_code_entry content: {stock_code_entry}")

            # Handle different data types
            if isinstance(stock_code_entry, str):
                try:
                    stock_codes = ast.literal_eval(stock_code_entry)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing stock codes for date {date.date()}: {e}")
                    return []
            elif isinstance(stock_code_entry, np.ndarray):
                # Convert numpy ndarray to list
                stock_codes = stock_code_entry.tolist()
            elif isinstance(stock_code_entry, list):
                stock_codes = stock_code_entry
            elif hasattr(stock_code_entry, "__iter__") and not isinstance(stock_code_entry, (str, bytes)):
                # For any other iterable types
                stock_codes = list(stock_code_entry)
            else:
                logger.error(f"Unexpected data type for stock codes: {type(stock_code_entry)}")
                return []

            # Convert all stock codes to strings
            stock_codes = [str(code) for code in stock_codes]

            # Verify that stock_codes is a list of strings
            if isinstance(stock_codes, list) and all(isinstance(code, str) for code in stock_codes):
                logger.info(f"Found {len(stock_codes)} stocks for date {date.date()}")
                return stock_codes
            else:
                logger.error(f"Stock codes are not a list of strings for date {date.date()}")
                return []
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

    previous_date = date - pd.Timedelta(days=1)
    previous_data = read_stock_data(stock_code, previous_date, csv_path)
    if previous_data.empty:
        logger.warning(f"No data for stock: {stock_code} on previous date: {previous_date.date()}")
        return {}
    previous_day_last_close = previous_data['Close'].iloc[-1]

    entry_time_str = config['entry_time']
    entry_time_min = int(entry_time_str.split(':')[0]) * 60 + int(entry_time_str.split(':')[1])

    equity, trades, holding_minutes = execute_short_strategy(
        data_np['Open'], data_np['Close'], data_np['VWAP'], data_np['open_time'],
        config['initial_cap'], config['fee_rate'], config['stop_loss_pct'],
        config['threshold'], entry_time_min, previous_day_last_close
    )

    performance = calculate_daily_performance(equity, trades)
    performance['Stock'] = stock_code
    performance['Date'] = date.date()
    performance['Equity Curve'] = equity  # Add equity curve to results
    performance['Holding Minutes'] = holding_minutes

    return performance


def backtest_wrapper(args):
    stock_code, date, config = args
    return backtest_single_stock(stock_code, date, config)



def backtest_multiple_stocks(stocks: list, date: datetime, config: dict) -> tuple:
    """
    Runs backtest for multiple stocks in parallel and returns both individual results and equity curves.
    """
    args_list = [(stock_code, date, config) for stock_code in stocks]
    
    with Pool(processes=config.get('num_processes', 4)) as pool:
        results = pool.map(backtest_wrapper, args_list)
    
    # Separate results and equity curves
    valid_results = []
    equity_curves = {}
    
    for result in results:
        if result and 'Stock' in result:
            stock_code = result['Stock']
            valid_results.append(result)
            if 'Equity Curve' in result:
                equity_curves[(result['Date'], stock_code)] = result['Equity Curve']
    
    return valid_results, equity_curves


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
        

def main(config_path: str, start_date_str: str, end_date_str: str, stock_list_file: str):
    """
    Main function to execute backtesting over a date range.
    """
    try:
        config = load_config(config_path)
        if not config:
            logger.error("Failed to load configuration")
            return

        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("Dates must be in YYYY-MM-DD format")
            return

        if start_date > end_date:
            logger.error("Start date must be before or equal to end date")
            return

        # Generate list of dates within the range
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        all_results = []
        all_equity_curves = {}

        for date in date_list:
            logger.info(f"Processing date: {date.date()}")

            stocks = get_stocks_for_date(stock_list_file, date)
            if not stocks:
                logger.warning(f"No stocks found for date: {date.date()}")
                continue

            # Get both individual results and equity curves
            results, equity_curves = backtest_multiple_stocks(stocks, date, config)

            if results:
                # Append daily results
                all_results.extend(results)

                # Merge equity curves
                all_equity_curves.update(equity_curves)
            else:
                logger.warning(f"No valid results from backtesting on date: {date.date()}")

        if all_results:
            # Process aggregated results
            results_df = pd.DataFrame(all_results)

            # Adjust decimal places to 6
            pd.options.display.float_format = '{:.6f}'.format

            # Ensure 'Date' column is datetime
            results_df['Date'] = pd.to_datetime(results_df['Date'])

            # Save the aggregated results to CSV
            output_directory = config.get('output_directory', '.')
            os.makedirs(output_directory, exist_ok=True)
            output_file = Path(output_directory) / f'backtest_results_{start_date.date()}_to_{end_date.date()}.csv'

            # Compute and output aggregated metrics
            aggregated_metrics = compute_aggregated_metrics(results_df)

            # Save combined results to one CSV file
            save_combined_results(results_df, aggregated_metrics, output_file)

            logger.info(f"Combined results saved to {output_file}")
        else:
            logger.warning("No valid results from backtesting over the date range")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise e
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock backtesting over a date range")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("start_date", help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("stock_list_file", help="Path to file containing stock codes and dates")
    args = parser.parse_args()

    try:
        main(args.config, args.start_date, args.end_date, args.stock_list_file)
    except Exception as e:
        logger.error(f"Program terminated with error: {e}")
        exit(1)
