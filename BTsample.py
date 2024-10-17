import pandas as pd
import numpy as np
from SR_idcts import BBands as SR_BBands
import matplotlib.pyplot as plt

def construct_data(data: pd.DataFrame, start_time: str = None, end_time: str = None) -> tuple:
    """
    Constructs and returns data arrays and signals required for analysis.
    
    Args:
        data (pd.DataFrame): DataFrame containing columns 'open', 'high', 'low', 'close', 
                             'funding_rate', and 'open_time'.
        start_time (str): Start time for filtering data (inclusive).
        end_time (str): End time for filtering data (exclusive).
                             
    Returns:
        tuple: Includes lengths, numpy arrays of price data, pandas series of price data, 
               and funding signal.
    """
    # Ensure 'open_time' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['open_time']):
        data['open_time'] = pd.to_datetime(data['open_time'])
    
    # Filter data based on start_time and end_time
    if start_time is not None:
        data = data[data['open_time'] >= pd.to_datetime(start_time)]
    if end_time is not None:
        data = data[data['open_time'] < pd.to_datetime(end_time)]
    # If both are None, data remains unchanged
    data.reset_index(drop=True, inplace=True)
    idx_len = len(data)
    converting_columns = ['open', 'high', 'low', 'close', 'volume', 'funding_rate']
    price_data_np = {col: data[col].to_numpy(dtype=float) for col in converting_columns}
    price_data_pd = {col: data[col] for col in converting_columns}
    
    open_time_np = data['open_time'].to_numpy(dtype='datetime64[m]')
    open_time_pd = data['open_time'].dt.floor('min')
    price_data_np['open_time'] = open_time_np
    price_data_pd['open_time'] = open_time_pd
    
    funding_signal = np.isin(data['open_time'].dt.strftime('%H:%M'), ['00:00', '08:00', '16:00'])
    return idx_len, price_data_np, price_data_pd, funding_signal


def generate_signals(close_pd: pd.Series, long_params: tuple, short_params: tuple) -> tuple:
    """
    Generates trading signals based on Bollinger Bands.
    
    Args:
        close_pd (pd.Series): Series of closing prices.
        long_params (tuple): Parameters for long BBands (period, upper bound, lower bound).
        short_params (tuple): Parameters for short BBands (period, upper bound, lower bound).
    
    Returns:
        tuple: Buy and sell signals.
    """
    long_period, long_ub, long_lb = long_params
    short_period, short_ub, short_lb = short_params

    long_SMA, long_UB, long_LB = SR_BBands(close_pd, long_period, long_ub, long_lb)
    short_SMA, short_UB, short_LB = SR_BBands(close_pd, short_period, short_ub, short_lb)
    
    signal_1 = (close_pd > long_UB).shift(1).to_numpy()
    signal_2 = (close_pd < long_LB).shift(1).to_numpy()
    signal_3 = (close_pd < short_LB).shift(1).to_numpy()
    signal_4 = (close_pd > short_UB).shift(1).to_numpy()
    return signal_1, signal_2, signal_3, signal_4


def execute_trading_strategy(idx_len: int, price_data_np: dict, price_data_pd: dict, 
                             funding_signal: np.ndarray, signals: tuple, 
                             fee_rate: float, initial_cap: float) -> np.ndarray:
    """
    Executes the trading strategy and calculates the equity curve.
    
    Args:
        idx_len (int): Length of the dataset.
        price_data_np (dict): Dictionary of numpy arrays for price data ('open', 'close', etc.).
        price_data_pd (dict): Dictionary of pandas series for price data.
        funding_signal (np.ndarray): Array indicating funding fee time points.
        signals (tuple): Tuple containing buy and sell signals.
        fee_rate (float): Transaction fee rate.
        initial_cap (float): Initial capital.
        
    Returns:
        np.ndarray: Array of equity values over time.
    """
    open_np = price_data_np['open']
    close_np = price_data_np['close']
    funding_rate_np = price_data_np['funding_rate']
    signal_1, signal_2, signal_3, signal_4 = signals

    cap = np.full(idx_len, np.nan, dtype=float)
    cap[0] = initial_cap
    equity = np.full(idx_len, np.nan, dtype=float)
    equity[0] = initial_cap
    long_position = np.zeros(idx_len, dtype=float)
    short_position = np.zeros(idx_len, dtype=float)
    
    short_entry_price = 0
    for i in range(0, idx_len - 1):
        # Long entry
        if cap[i] >= 0 and long_position[i-1] <= 0 and signal_1[i]:
            long_entry_amout = cap[i]
            long_entry_price = open_np[i]
            long_position[i] = long_entry_amout / (long_entry_price * (1 + fee_rate))
            cap[i] -= long_entry_amout
        # Long exit
        elif long_position[i] > 0 and signal_2[i]:
            long_exit_price = open_np[i]
            cap[i] += long_position[i] * (long_exit_price * (1 - fee_rate))
            long_position[i] = 0
        
        # Short entry
        if cap[i] >= 0 and short_position[i] <= 0 and signal_3[i]:
            short_entry_price = open_np[i]
            short_position[i] = cap[i] / (short_entry_price * (1 + fee_rate))
            cap[i] -= cap[i]
        # Short exit
        elif short_position[i] > 0 and signal_4[i]:
            short_exit_price = open_np[i]
            cap[i] += short_position[i] * (2 * short_entry_price - short_exit_price * (1 + fee_rate))
            short_position[i] = 0
        
            
        # Update equity including unrealized P&L
        equity[i] = cap[i]
        if long_position[i] > 0:
            equity[i] += long_position[i] * close_np[i]
        if short_position[i] > 0:     
            equity[i] += short_position[i] * (2 * short_entry_price - close_np[i])
        
        # # Deduct funding fees if applicable
        # if funding_signal[i]:
        #     funding_fee = 0
        #     if long_position[i] > 0:
        #         funding_fee += long_position[i] * open_np[i] * funding_rate_np[i]
        #     if short_position[i] > 0:
        #         funding_fee -= short_position[i] * open_np[i] * funding_rate_np[i]
        #     equity[i] -= funding_fee
        #     cap[i] -= funding_fee

        # Carry forward capital and positions
        cap[i + 1] = cap[i]
        long_position[i + 1] = long_position[i]
        short_position[i + 1] = short_position[i]

    # Final equity update for the last
    equity[idx_len-1] = cap[idx_len-1] \
        + long_position[idx_len-1] * close_np[idx_len-1] \
        + short_position[idx_len-1] * (2 * short_entry_price - close_np[idx_len-1])
    return equity

def plot_results(open_time_np: np.ndarray, equity: np.ndarray, close_np: np.ndarray):
    """
    Plots the equity curve vs Buy&Hold, including the underwater plot,
    marking new high points and highlighting the top five longest drawdown periods.
    
    Args:
        open_time_np (np.ndarray): Array of open times.
        equity (np.ndarray): Array of equity values.
        close_np (np.ndarray): Array of close prices.
    """
    equity_ret = equity / equity[0] - 1
    close_ret = close_np / close_np[0] - 1

    # Calculate drawdown
    cummax_equity = np.maximum.accumulate(equity)
    drawdown = (cummax_equity - equity) / cummax_equity

    # Identify new high points
    new_highs = (equity == cummax_equity)

    # Find drawdown periods
    dd_periods = drawdown > 0
    # Find drawdown start and end indices
    dd_starts = np.where((~dd_periods[:-1]) & (dd_periods[1:]))[0] + 1
    dd_ends = np.where((dd_periods[:-1]) & (~dd_periods[1:]))[0] + 1

    # If drawdown starts at the beginning
    if dd_periods[0]:
        dd_starts = np.insert(dd_starts, 0, 0)

    # If drawdown ends after the last data point
    if dd_periods[-1]:
        dd_ends = np.append(dd_ends, len(drawdown) - 1)

    # Calculate maximum drawdowns and durations
    mdds = []
    for start, end in zip(dd_starts, dd_ends):
        mdd = np.max(drawdown[start:end+1])
        duration = (open_time_np[end] - open_time_np[start]).astype('timedelta64[m]').astype(int)
        mdds.append({'start': start, 'end': end, 'mdd': mdd, 'duration': duration})

    # Sort by duration (longest first)
    mdds_sorted = sorted(mdds, key=lambda x: x['duration'], reverse=True)
    top_mdds = mdds_sorted[:5]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(open_time_np, equity_ret, label='Equity Curve', linewidth=1)
    ax1.plot(open_time_np, close_ret, label='Buy & Hold', linewidth=1)

    ax1.scatter(open_time_np[new_highs], equity_ret[new_highs], color='green', marker='^', label='New Highs')

    # Highlight top five longest drawdown areas
    for mdd_info in top_mdds:
        start_idx = mdd_info['start']
        end_idx = mdd_info['end']
        ax1.axvspan(open_time_np[start_idx], open_time_np[end_idx], color='red', alpha=0.2)

    ax1.set_title('Equity Curve vs Buy&Hold')
    ax1.set_ylabel('Normalized Return')
    ax1.legend()
    ax1.grid(True)

    # Underwater plot
    ax2.fill_between(open_time_np, -drawdown, 0, color='blue', alpha=0.5)
    ax2.set_title('Underwater Plot (Drawdown)')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Time')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


#%% Main Execution
if __name__ == "__main__":
    # Initialize
    data = pd.read_pickle('BTCUSDT_perp_data/BTCUSDT_15m.pkl')
    data.reset_index(drop=True, inplace=True)
    start_time = '2022-01-01 00:00:00'
    end_time = '2024-01-01 00:00:00'

    # Construct data arrays and signals
    idx_len, price_data_np, price_data_pd, funding_signal = construct_data(data, start_time, end_time)
    price_data = price_data_pd['close']

    # Define strategy parameters
    fee_rate = 0.0005
    initial_cap = 10000.0
    long_params = (1800, 1.0, 0.0)
    short_params = (100, 1.0, 0.0)

    # Generate signals
    signals = generate_signals(price_data, long_params, short_params)

    