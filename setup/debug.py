import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import yaml
from daily_bt import (
    load_config, 
    read_stock_data, 
    construct_data, 
    execute_short_strategy,
    calculate_daily_performance,
    backtest_single_stock,
    get_stocks_for_date
)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Creates sample data for testing"""
    # Create a sample date range for one trading day
    date_range = pd.date_range('2023-11-16 09:30:00', '2023-11-16 16:00:00', freq='1min')
    
    # Create sample price data
    data = pd.DataFrame({
        'ts': date_range,  # Add ts column explicitly
        'Open': np.random.uniform(100, 110, len(date_range)),
        'High': np.random.uniform(105, 115, len(date_range)),
        'Low': np.random.uniform(95, 105, len(date_range)),
        'Close': np.random.uniform(100, 110, len(date_range)),
        'Volume': np.random.randint(1000, 10000, len(date_range))
    })
    
    # Ensure High is highest and Low is lowest
    data['High'] = np.maximum(data[['Open', 'Close', 'High']].max(axis=1), data['High'])
    data['Low'] = np.minimum(data[['Open', 'Close', 'Low']].min(axis=1), data['Low'])
    
    return data

def create_sample_config():
    """Creates sample configuration for testing"""
    config = {
        'ma_period': 5,
        'initial_cap': 100000,
        'fee_rate': 0.001,
        'stop_loss_pct': 0.04,
        'threshold': 1.01,
        'entry_time': '09:06',
        'csv_path': './test_data',
        'output_directory': './test_output'
    }
    return config

def test_load_config():
    """Test configuration loading"""
    logger.info("Testing load_config function...")
    
    # Create a temporary config file
    test_config = create_sample_config()
    with open('test_config.yaml', 'w') as f:
        yaml.dump(test_config, f)
    
    # Test loading
    loaded_config = load_config('test_config.yaml')
    assert loaded_config == test_config, "Config loading failed"
    logger.info("Config loading test passed")

def test_read_stock_data():
    """Test stock data reading"""
    logger.info("Testing read_stock_data function...")
    
    try:
        # Create test directory and sample data
        Path('./test_data').mkdir(exist_ok=True)
        sample_data = create_sample_data()
        
        # Save with ts column
        sample_data.to_csv('./test_data/TEST.csv', index=False)
        
        # Test reading
        date = datetime.strptime('2023-11-16', '%Y-%m-%d')
        data = read_stock_data('TEST', date, './test_data')
        
        assert not data.empty, "Stock data reading failed"
        assert 'Open' in data.columns, "Missing Open column"
        assert 'Close' in data.columns, "Missing Close column"
        assert 'High' in data.columns, "Missing High column"
        assert 'Low' in data.columns, "Missing Low column"
        assert 'Volume' in data.columns, "Missing Volume column"
        
        logger.info("Stock data reading test passed")
    except Exception as e:
        logger.error(f"Error in test_read_stock_data: {str(e)}")
        raise

def test_construct_data():
    """Test data construction"""
    logger.info("Testing construct_data function...")
    
    sample_data = create_sample_data()
    sample_data.set_index('ts', inplace=True)
    data_dict = construct_data(sample_data, ma_period=5)
    
    required_keys = ['Open', 'Close', 'High', 'Low', 'Volume', 'MA', 'VWAP', 'open_time']
    assert all(key in data_dict for key in required_keys), "Missing required keys in constructed data"
    logger.info("Data construction test passed")

def test_execute_short_strategy():
    """Test short strategy execution"""
    logger.info("Testing execute_short_strategy function...")
    
    sample_data = create_sample_data()
    sample_data.set_index('ts', inplace=True)
    data_dict = construct_data(sample_data, ma_period=5)
    
    equity, trades = execute_short_strategy(
        data_dict['Open'],
        data_dict['Close'],
        data_dict['VWAP'],
        data_dict['open_time'],
        initial_cap=100000,
        fee_rate=0.001,
        stop_loss_pct=0.04,
        threshold=1.01,
        entry_time_min=546,  # 9:06 AM
        previous_day_last_close=100.0
    )
    
    assert len(equity) == len(sample_data), "Equity array length mismatch"
    assert isinstance(trades, int), "Trades count should be integer"
    logger.info("Short strategy execution test passed")

def test_calculate_daily_performance():
    """Test performance calculation"""
    logger.info("Testing calculate_daily_performance function...")
    
    # Create sample equity curve
    equity = np.array([100000, 100100, 100200, 100150, 100300])
    trades = 2
    
    performance = calculate_daily_performance(equity, trades)
    required_metrics = ['Total Trades', 'Total Return', 'Max Drawdown', 'Win Rate', 'Daily Volatility']
    
    assert all(metric in performance for metric in required_metrics), "Missing required performance metrics"
    logger.info("Performance calculation test passed")

def test_backtest_single_stock():
    """Test single stock backtesting"""
    logger.info("Testing backtest_single_stock function...")
    
    config = create_sample_config()
    date = datetime.strptime('2023-11-16', '%Y-%m-%d')
    
    # Create test data for current day
    sample_data = create_sample_data()
    Path('./test_data').mkdir(exist_ok=True)
    sample_data.to_csv('./test_data/TEST.csv', index=False)
    
    # Create previous day's data
    prev_date = date - pd.Timedelta(days=1)
    sample_data['ts'] = sample_data['ts'] - pd.Timedelta(days=1)
    sample_data.to_csv('./test_data/TEST.csv', mode='a', header=False, index=False)
    
    result = backtest_single_stock('TEST', date, config)
    assert isinstance(result, dict), "Backtest result should be a dictionary"
    logger.info("Single stock backtest test passed")

def test_get_stocks_for_date():
    """Test stock list retrieval"""
    logger.info("Testing get_stocks_for_date function...")
    
    # Create sample stock list file
    date = datetime.strptime('2023-11-16', '%Y-%m-%d')
    stock_list_df = pd.DataFrame({
        'index': [date.date()],
        'stock_list': [str(['TEST1', 'TEST2', 'TEST3'])]
    })
    stock_list_df.to_feather('test_stock_list.feather')
    
    stocks = get_stocks_for_date('test_stock_list.feather', date)
    assert isinstance(stocks, list), "Stock list should be a list"
    assert len(stocks) > 0, "Stock list should not be empty"
    logger.info("Stock list retrieval test passed")

def main():
    """Run all tests"""
    logger.info("Starting debug tests...")
    
    try:
        test_load_config()
        test_read_stock_data()
        test_construct_data()
        test_execute_short_strategy()
        test_calculate_daily_performance()
        test_backtest_single_stock()
        test_get_stocks_for_date()
        
        logger.info("All tests completed successfully!")
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during testing: {str(e)}")
    finally:
        # Cleanup test files
        import shutil
        if Path('./test_data').exists():
            shutil.rmtree('./test_data')
        if Path('./test_output').exists():
            shutil.rmtree('./test_output')
        if Path('test_config.yaml').exists():
            Path('test_config.yaml').unlink()
        if Path('test_stock_list.feather').exists():
            Path('test_stock_list.feather').unlink()

if __name__ == "__main__":
    main()