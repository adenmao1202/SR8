import pandas as pd

def SMA(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA) over a specified period.

    Parameters:
    prices (pd.Series): A Pandas Series containing the price data (e.g., daily, weekly).
    period (int): The window size for calculating the SMA.

    Returns:
    pd.Series: A Pandas Series containing the SMA values, with the same index as the input.
    """
    # Validate input parameters
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a Pandas Series.")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer.")

    # Calculate the SMA using a rolling mean
    return prices.rolling(window=period).mean()


def BBands(prices: pd.Series, period: int, ub_mult: float, lb_mult: float) -> tuple:
    """
    Calculate the Bollinger Bands (BBands) using a Simple Moving Average (SMA)
    for the middle band, with customizable multipliers for the upper and lower bands.

    Parameters:
    prices (pd.Series): A Pandas Series containing the price data (e.g., daily, weekly).
    period (int): The window size for calculating the SMA and standard deviation.
    ub_mult (float): The multiplier applied to the standard deviation for the upper band.
    lb_mult (float): The multiplier applied to the standard deviation for the lower band.

    Returns:
    tuple: A tuple containing three Pandas Series:
        - middle_band (pd.Series): The SMA over the specified period.
        - upper_band (pd.Series): The upper Bollinger Band (SMA + ub_mult * std).
        - lower_band (pd.Series): The lower Bollinger Band (SMA - lb_mult * std).
    """
    # Validate input parameters
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a Pandas Series.")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer.")
    if not isinstance(ub_mult, (int, float)):
        raise ValueError("ub_mult must be a positive number.")
    if not isinstance(lb_mult, (int, float)):
        raise ValueError("lb_mult must be a positive number.")
    
    middle_band = SMA(prices, period)
    bound = prices.rolling(window=period).std()
    
    # Calculate the upper and lower bands
    upper_band = middle_band + ub_mult * bound
    lower_band = middle_band - lb_mult * bound
    
    return middle_band, upper_band, lower_band



def EMA(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) over a specified period.
    
    Parameters:
    prices (pd.Series): A Pandas Series containing the price data (e.g., daily, weekly).
    period (int): The smoothing period for calculating the EMA.
    
    Returns:
    pd.Series: A Pandas Series containing the EMA values.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a Pandas Series.")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer.")
    
    return prices.ewm(span=period, adjust=False).mean()

def TR(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Calculate the True Range (TR) for the given high, low, and close prices.

    Parameters:
    high (pd.Series): A Pandas Series containing high prices.
    low (pd.Series): A Pandas Series containing low prices.
    close (pd.Series): A Pandas Series containing close prices.

    Returns:
    pd.Series: A Pandas Series containing the True Range (TR) values.
    """
    if not all(isinstance(series, pd.Series) for series in [high, low, close]):
        raise TypeError("high, low, and close must be Pandas Series.")

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Average True Range (ATR) over a specified period for given high, low, and close prices.

    Parameters:
    high (pd.Series): A Pandas Series containing high prices.
    low (pd.Series): A Pandas Series containing low prices.
    close (pd.Series): A Pandas Series containing close prices.
    period (int): The period over which to calculate the ATR (commonly 14).

    Returns:
    pd.Series: A Pandas Series containing the ATR values.
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer.")
    
    tr_values = TR(high, low, close)
    return tr_values.rolling(window=period, min_periods=1).mean()