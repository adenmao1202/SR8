import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas_market_calendars as mcal
from concurrent.futures import ProcessPoolExecutor
import os


def read_stock_data(stock_code: str, file_path: Path) -> pd.DataFrame:
    """Read and resample stock data from CSV."""
    stock_df = pd.read_csv(
        file_path,
        usecols=['ts', 'Close', 'High', 'Volume'],
        parse_dates=['ts'],
        dtype={'Close': 'float64', 'High': 'float64', 'Volume': 'int64'}  # Ensuring proper data types
    )
    stock_df.set_index('ts', inplace=True)
    
    # Resample to daily frequency and aggregate
    stock_df = stock_df.resample('D').agg({
        'Close': 'last',
        'High': 'max',
        'Volume': 'sum'
    })
    stock_df['stock_code'] = stock_code
    return stock_df.reset_index()


def read_target_stocks(feather_file_path: Path) -> List[str]:
    """Read list of target stocks from a Feather file."""
    df = pd.read_feather(feather_file_path)
    return df['stock_code'].astype(str).tolist()


def load_and_pivot_stock_data(
    target_stocks: List[str], data_folder: Path, business_days: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and pivot all stock data into separate DataFrames."""
    
    stock_data_list = []
    
    # Use ProcessPoolExecutor to parallelize I/O operations
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        # for csv_file in data_folder.glob('*.csv'):
        #     stock_code = csv_file.stem
        for stock_code in target_stocks:
            file_path = data_folder / f'{stock_code}.csv'
            if not file_path.exists():
                continue
            futures.append(executor.submit(read_stock_data, stock_code, file_path))
        
        # Collect the results
        for future in futures:
            try:
                stock_data_list.append(future.result())
            except Exception as e:
                print(f"Error processing file: {e}")
    
    
    # # If don't use ProcessPoolExecuto
    # for stock_code in target_stocks:
    #     file_path = data_folder / f'{stock_code}.csv'
    #     if not file_path.exists():
    #         continue
    #     stock_df = read_stock_data(stock_code, file_path)
    #     stock_data_list.append(stock_df)
    
    if not stock_data_list:
        return None, None, None

    combined_df = pd.concat(stock_data_list, ignore_index=True)

    # Pivot data into individual DataFrames for Close, High, and Volume
    close_df = combined_df.pivot(index='ts', columns='stock_code', values='Close').reindex(business_days)
    high_df = combined_df.pivot(index='ts', columns='stock_code', values='High').reindex(business_days)
    volume_df = combined_df.pivot(index='ts', columns='stock_code', values='Volume').reindex(business_days)

    return close_df, high_df, volume_df


def compute_indicators(
        close_df: pd.DataFrame,
        high_df: pd.DataFrame, 
        volume_df: pd.DataFrame, 
        offset_days: int, 
        offset_days_2: int,
        offset_days_3: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute percentage change, new highs, and volume criteria indicators."""
    # Using shift to calculate changes
    close_T_minus_1 = close_df.shift(1)
    close_T_minus_offset = close_df.shift(offset_days)
    pct_change_df = (close_T_minus_offset / close_T_minus_1) - 1

    # Calculate max high over the rolling window
    max_high_df = high_df.shift(2).rolling(window=offset_days_2).max()
    is_new_high_df = close_T_minus_1 >= max_high_df

    # Calculate volume criteria
    mean_volume_df = volume_df.shift(2).rolling(window=offset_days_3).mean()
    volume_T_minus_1 = volume_df.shift(1)
    volume_criteria_df = (volume_T_minus_1 >= 3 * mean_volume_df) & (volume_T_minus_1 > 1000)

    return pct_change_df, is_new_high_df, volume_criteria_df


def process_dates(business_days: pd.DatetimeIndex, pct_change_df: pd.DataFrame, is_new_high_df: pd.DataFrame, volume_criteria_df: pd.DataFrame, top_n: int, offset_days: int) -> pd.DataFrame:
    """Process each date to compute top stocks and return the result DataFrame."""
    result_df = pd.DataFrame(index=business_days, columns=['stock_list'])
    result_df['stock_list'] = [[] for _ in range(len(result_df))]

    for idx in range(offset_days + 2, len(business_days)):
        date = business_days[idx]
        pct_change_series = pct_change_df.iloc[idx].replace([np.inf, -np.inf], np.nan).dropna()

        if pct_change_series.empty:
            continue

        # Get top N stocks by percentage change
        top_stocks = pct_change_series.nlargest(top_n).index.tolist()
        is_new_high_series = is_new_high_df.iloc[idx][top_stocks]
        volume_criteria_series = volume_criteria_df.iloc[idx][top_stocks]
        selected_stocks = is_new_high_series[is_new_high_series & volume_criteria_series].index.tolist()

        if selected_stocks:
            result_df.at[date, 'stock_list'] = selected_stocks

    return result_df



if __name__ == "__main__":
    target_stocks_path = Path('/Users/mouyasushi/Desktop/sr8_op_file/target_stocks.feather')
    data_folder = Path('/Users/mouyasushi/k_data/永豐')

    target_stocks = read_target_stocks(target_stocks_path)

    start_date_str, end_date_str = '2022-10-14', '2024-10-14'
    offset_days = 3
    offset_days_2 = 3
    offset_days_3 = 3
    top_n = 200

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    business_days = mcal.get_calendar('XTAI').schedule(start_date=start_date, end_date=end_date).index

    close_df, high_df, volume_df = load_and_pivot_stock_data(target_stocks, data_folder, business_days)

    pct_change_df, is_new_high_df, volume_criteria_df = compute_indicators(close_df, high_df, volume_df, offset_days, offset_days_2, offset_days_3)
    result_df = process_dates(business_days, pct_change_df, is_new_high_df, volume_criteria_df, top_n, offset_days)

    
    # 过滤并保存为 Feather 文件
    filtered_result_df = result_df[result_df['stock_list'].apply(bool)]
    #
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    print(filtered_result_df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    #
    output_path = Path('/Users/mouyasushi/Desktop/sr8_op_file/target_stocks.feather')
    filtered_result_df.reset_index().to_feather(output_path)
    
    print(f"Filtered result saved to {output_path}")