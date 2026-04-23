"""Data loader — fetch historical OHLCV from Yahoo Finance (free)."""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = "/tmp/trading_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Indian market tickers available on Yahoo Finance
PRESET_TICKERS = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "BTC": "BTC-USD",
}


def load(
    ticker: str,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data. Returns DataFrame with lowercase column names.
    ticker can be a preset name (e.g. 'NIFTY50') or a raw Yahoo symbol.
    """
    symbol = PRESET_TICKERS.get(ticker, ticker)
    cache_key = f"{symbol}_{start}_{end}_{interval}".replace("/", "-").replace(":", "")
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")

    if use_cache and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df

    raw = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data returned for {symbol}. Check ticker or date range.")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()

    if use_cache:
        df.to_parquet(cache_path)

    return df


def get_train_test(
    ticker: str,
    train_start: str = "2020-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2025-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch separate train and test datasets (no leakage)."""
    train = load(ticker, train_start, train_end)
    test = load(ticker, test_start, test_end)
    return train, test


def available_tickers() -> list[str]:
    return list(PRESET_TICKERS.keys())