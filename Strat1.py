import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def read_stock_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    required_columns = ['ts', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file {file_path} must have columns: {', '.join(required_columns)}")
    
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    
    if df.index.inferred_freq != '1min':
        df = df.resample('1min').last().ffill()
    
    return df

def construct_data(data: pd.DataFrame, start_time: str = None, end_time: str = None) -> Tuple[int, Dict[str, np.ndarray], Dict[str, pd.Series]]:
    if start_time is not None:
        data = data[data.index >= pd.to_datetime(start_time)]
    if end_time is not None:
        data = data[data.index < pd.to_datetime(end_time)]

    data.reset_index(inplace=True)
    data.rename(columns={'ts': 'open_time'}, inplace=True)
    idx_len = len(data)
    
    converting_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    price_data_np = {col: data[col].to_numpy(dtype=float) for col in converting_columns}
    price_data_pd = {col: data[col] for col in converting_columns}
    
    open_time_np = data['open_time'].to_numpy(dtype='datetime64[m]')
    open_time_pd = data['open_time'].dt.floor('min')
    price_data_np['open_time'] = open_time_np
    price_data_pd['open_time'] = open_time_pd
    
    # Calculate moving average
    ma_period = 20  # You can adjust this
    price_data_np['MA'] = np.convolve(price_data_np['Close'], np.ones(ma_period), 'valid') / ma_period
    
    # Calculate intraday buying/selling pressure
    price_data_np['BuyPressure'] = price_data_np['Volume'] * (price_data_np['Close'] > price_data_np['Open']).astype(int)
    price_data_np['SellPressure'] = price_data_np['Volume'] * (price_data_np['Close'] < price_data_np['Open']).astype(int)
    
    return idx_len, price_data_np, price_data_pd

def execute_short_strategy(idx_len: int, price_data_np: Dict[str, np.ndarray], 
                           threshold_range: Tuple[float, float], entry_range: Tuple[int, int], 
                           fee_rate: float, initial_cap: float,
                           stop_loss_pct: float, take_profit_pct: float) -> Tuple[np.ndarray, int]:
    open_np = price_data_np['Open']
    high_np = price_data_np['High']
    low_np = price_data_np['Low']
    close_np = price_data_np['Close']
    ma_np = price_data_np['MA']
    buy_pressure_np = price_data_np['BuyPressure']
    sell_pressure_np = price_data_np['SellPressure']
    open_time_np = price_data_np['open_time']
    
    cap = np.full(idx_len, np.nan, dtype=float)
    cap[0] = initial_cap
    equity = np.full(idx_len, np.nan, dtype=float)
    equity[0] = initial_cap
    short_position = np.zeros(idx_len, dtype=float)
    short_entry_price = 0
    trade_count = 0
    
    # Get daily first K-bar index (8:45 AM)
    first_kbars_idx = np.where(pd.to_datetime(open_time_np).strftime('%H:%M') == '08:45')[0]

    def check_stop_loss(i, entry_price):
        # Stop loss condition 1: Price crosses above MA
        if close_np[i] > ma_np[i] and close_np[i-1] <= ma_np[i-1]:
            return True
        
        # Stop loss condition 2: -4% loss (adjustable)
        if close_np[i] >= entry_price * (1 + stop_loss_pct):
            return True
        
        # Stop loss condition 3: Intraday selling pressure turns to buying pressure
        if i > 0 and sell_pressure_np[i-1] > buy_pressure_np[i-1] and sell_pressure_np[i] <= buy_pressure_np[i]:
            return True
        
        return False

    def check_take_profit(current_price, entry_price, take_profit_pct):
        return current_price <= entry_price * (1 - take_profit_pct)

    # Threshold loop
    for threshold in np.arange(threshold_range[0], threshold_range[1], 0.01):
        # Entry K-bar loop
        for entry_k in range(entry_range[0], entry_range[1]):
            for i in range(entry_k, idx_len - 1):
                # Check if it's the first K-bar of the day (8:45 AM)
                if i in first_kbars_idx:
                    # Check for gap up (open > previous close by threshold) on the first K-bar of the day
                    if open_np[i] / close_np[i-1] > 1 + threshold:
                        # Short entry if no position
                        if cap[i] > 0 and short_position[i-1] == 0:
                            short_entry_price = open_np[i]
                            short_position[i] = cap[i] / (short_entry_price * (1 + fee_rate))
                            cap[i] = 0  # Use all capital to enter the short
                            trade_count += 1

                # Check for stop loss or take profit
                if short_position[i] > 0:
                    if check_stop_loss(i, short_entry_price) or check_take_profit(low_np[i], short_entry_price, take_profit_pct):
                        exit_price = open_np[i+1]  # Exit at next bar's open
                        cap[i+1] = short_position[i] * (2 * short_entry_price - exit_price * (1 + fee_rate))
                        short_position[i+1] = 0
                        continue

                # Exit 1/3 of position at specified times
                if short_position[i] > 0:
                    if pd.to_datetime(open_time_np[i]).strftime('%H:%M') in ['11:00', '13:00', '13:30']:
                        short_exit_price = open_np[i]
                        cap[i] += (short_position[i] / 3) * (2 * short_entry_price - short_exit_price * (1 + fee_rate))
                        short_position[i] *= (2/3)  # Reduce position by 1/3
            
                # Update equity
                equity[i] = cap[i]
                if short_position[i] > 0:
                    equity[i] += short_position[i] * (2 * short_entry_price - close_np[i])

                # Carry forward positions and capital
                cap[i + 1] = cap[i]
                short_position[i + 1] = short_position[i]

            # Final equity update
            equity[idx_len-1] = cap[idx_len-1] + short_position[idx_len-1] * (2 * short_entry_price - close_np[idx_len-1])
    
    return equity, trade_count

def calculate_performance_metrics(equity: np.ndarray, price_data_np: Dict[str, np.ndarray], trade_count: int) -> Dict[str, float]:
    returns = np.diff(equity) / equity[:-1]
    buy_hold_returns = np.diff(price_data_np['Close']) / price_data_np['Close'][:-1]
    
    total_return = (equity[-1] / equity[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(equity)) - 1
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
    
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

def run_backtest(file_path: str, start_time: str, end_time: str, fee_rate: float, initial_cap: float,
                 threshold_range: Tuple[float, float], entry_range: Tuple[int, int],
                 stop_loss_pct: float, take_profit_pct: float) -> Dict[str, float]:
    # Read data
    data = read_stock_data(file_path)
    
    # Construct data arrays and signals
    idx_len, price_data_np, price_data_pd = construct_data(data, start_time, end_time)
    
    # Execute strategy
    equity, trade_count = execute_short_strategy(idx_len, price_data_np, threshold_range, entry_range, 
                                    fee_rate, initial_cap, stop_loss_pct, take_profit_pct)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(equity, price_data_np, trade_count)
    
    return metrics

def main():
    data_folder = Path('/Users/mouyasushi/k_data/永豐')
    start_time = '2021-10-14 09:06:00'
    end_time = '2024-10-14 13:30:00'
    fee_rate = 0.0005
    initial_cap = 10000.0
    threshold_range = (0.01, 0.04)
    entry_range = (5, 15)
    stop_loss_pct = 0.04
    take_profit_pct = 0.03

    all_results = []

    for file in data_folder.glob('*.csv'):
        print(f"Processing {file.name}...")
        try:
            metrics = run_backtest(str(file), start_time, end_time, fee_rate, initial_cap,
                                   threshold_range, entry_range, stop_loss_pct, take_profit_pct)
            metrics['File'] = file.name
            all_results.append(metrics)
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")

    # Combine all results into a DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate average metrics across all stocks
    average_metrics = results_df.drop('File', axis=1).mean()
    average_metrics['File'] = 'Average'

    # Add average metrics to the results
    results_df = results_df.append(average_metrics, ignore_index=True)

    # Sort results by Total Return
    results_df = results_df.sort_values('Total Return', ascending=False)

    # Display results
    print("\nBacktesting Results:")
    print(results_df.to_string(index=False))

    # Save results to CSV
    output_file = data_folder / 'backtest_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()