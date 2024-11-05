import pandas as pd
import numpy as np
from numba import njit
import yaml
import os
import logging
from concurrent.futures import ProcessPoolExecutor
import traceback

# ... 其他导入和常量定义 ...

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"配置加载成功: {config}")  # 新增：打印加载的配置
    return config

def read_stock_data(file_path: str, config: dict) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    if df.index.inferred_freq != '1min':
        df = df.resample('1min').last().ffill()
        print(f"数据已重采样到1分钟频率: {file_path}")  # 新增：打印重采样信息
    df['returns'] = df['close'].pct_change()
    print(f"数据读取成功，形状: {df.shape}")  # 新增：打印数据形状
    return df

@njit
def calculate_indicators(close, high, low, volume, window):
    # ... 指标计算逻辑 ...
    return sma, rsi, macd, signal, cci

def apply_strategy(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # ... 策略应用逻辑 ...
    df['position'] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    print(f"策略应用完成，信号数量: 买入 {buy_signal.sum()}, 卖出 {sell_signal.sum()}")  # 新增：打印信号数量
    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    # ... 收益计算逻辑 ...
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
    print(f"收益计算完成，最终累积收益: {df['cumulative_returns'].iloc[-1]:.2%}")  # 新增：打印最终累积收益
    return df

def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    # ... 性能指标计算逻辑 ...
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
    print(f"性能指标计算完成: {metrics}")  # 新增：打印计算的性能指标
    return metrics

def process_file(file_path: str, config: dict) -> dict:
    try:
        df = read_stock_data(file_path, config)
        df = apply_strategy(df, config)
        df = calculate_returns(df)
        metrics = calculate_performance_metrics(df)
        print(f"文件处理完成: {file_path}")  # 新增：打印文件处理完成信息
        return metrics
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")  # 新增：打印错误信息
        return None

def process_batch(file_paths: list, config: dict) -> list:
    results = []
    for file_path in file_paths:
        result = process_file(file_path, config)
        if result:
            results.append(result)
    print(f"批处理完成，处理文件数: {len(results)}")  # 新增：打印批处理完成信息
    return results

def main(config_path: str):
    config = load_config(config_path)
    data_dir = config['data_dir']
    output_file = config['output_file']
    batch_size = config['batch_size']

    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    all_results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i+batch_size]
            results = executor.submit(process_batch, batch, config).result()
            all_results.extend(results)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_file, index=False)
        print(f"结果已保存到 {output_file}")  # 新增：打印结果保存信息
        
        average_metrics = df_results.mean()
        print(f"平均性能指标:\n{average_metrics}")  # 新增：打印平均性能指标
    else:
        print("没有生成任何结果")  # 新增：打印无结果信息

if __name__ == "__main__":
    config_path = "config.yaml"
    main(config_path)
