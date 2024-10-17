import numpy as np
from scipy.optimize import minimize
from batchVersion import execute_short_strategy_optimized, calculate_performance_metrics, read_stock_data, construct_data

def calmar_ratio(equity: np.ndarray):
    """
    Calculate the Calmar Ratio from the equity curve.
    
    Args:
        equity (np.ndarray): Equity curve over time.
        
    Returns:
        float: The Calmar Ratio (annualized return divided by maximum drawdown).
    """
    max_drawdown = np.max(1 - equity / np.maximum.accumulate(equity))
    total_return = (equity[-1] / equity[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(equity)) - 1  # 假設一年有252個交易日
    if max_drawdown == 0:
        return 0  # 避免除以零
    return annualized_return / max_drawdown

def objective(params, *args):
    """
    The objective function to maximize Calmar Ratio.
    
    Args:
        params (list): The parameters being optimized (threshold, entry_k).
        args: Additional arguments for the backtesting function.
    
    Returns:
        float: The negative of the Calmar Ratio (we minimize this in SciPy).
    """
    threshold, entry_k = params
    equity, trade_count = execute_short_strategy_optimized(*args, threshold, entry_k)
    return -calmar_ratio(equity)  # 我們希望最大化 Calmar Ratio，所以返回負值

def optimize_params(data_file: str, config: dict):
    """
    Run the parameter optimization to maximize Calmar Ratio.
    
    Args:
        data_file (str): The path to the stock data file.
        config (dict): Configuration parameters.
    
    Returns:
        dict: The best parameters and corresponding performance metrics.
    """
    # 加載股票數據
    data = read_stock_data(data_file, config)
    idx_len, price_data_np = construct_data(data, config['start_time'], config['end_time'], config)
    
    # 設置初始參數和參數範圍
    initial_params = [0.01, 10]  # 預設的 threshold 和 entry_k 初始值
    bounds = [(0.01, 1.0), (0, 100)]  # 設定門檻和 entry_k 的範圍

    # 執行參數優化
    res = minimize(objective, initial_params, args=(price_data_np['Open'], price_data_np['High'], price_data_np['Low'],
                                                    price_data_np['Close'], price_data_np['MA'],
                                                    price_data_np['BuyPressure'], price_data_np['SellPressure'],
                                                    price_data_np['open_time'], config['initial_cap'],
                                                    config['fee_rate'], config['stop_loss_pct'], config['take_profit_pct'],
                                                    config['entry_time'], config['exit_times']),
                   bounds=bounds)
    
    # 得到最佳參數
    best_params = res.x
    best_equity, trade_count = execute_short_strategy_optimized(price_data_np['Open'], price_data_np['High'],
                                                                price_data_np['Low'], price_data_np['Close'],
                                                                price_data_np['MA'], price_data_np['BuyPressure'],
                                                                price_data_np['SellPressure'], price_data_np['open_time'],
                                                                config['initial_cap'], config['fee_rate'],
                                                                config['stop_loss_pct'], config['take_profit_pct'],
                                                                best_params[0], int(best_params[1]),
                                                                config['entry_time'], config['exit_times'])
    
    # 計算績效指標
    metrics = calculate_performance_metrics(best_equity, price_data_np, trade_count, config)
    
    return {
        'Best Parameters': best_params,
        'Calmar Ratio': calmar_ratio(best_equity),
        'Performance Metrics': metrics
    }

if __name__ == "__main__":
    # 在此處加載 config 文件並運行優化
    config_path = "config.yaml"
    config = load_config(config_path)
    
    data_file = "path/to/your/stock_data.csv"
    result = optimize_params(data_file, config)
    
    print(f"最佳參數: {result['Best Parameters']}")
    print(f"Calmar Ratio: {result['Calmar Ratio']}")
    print(f"績效指標: {result['Performance Metrics']}")
