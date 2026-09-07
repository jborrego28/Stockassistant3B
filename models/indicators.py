import numpy as np
import pandas as pd


FEATURES = [
    "sma_5",
    "sma_20",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_histogram",
    "momentum",
    "rsi",
    "atr",
    "volume_ratio",
    "high_20",
    "low_20",
]


def calculate_indicators(history):
    data = history.copy()

    close = pd.to_numeric(data["Close"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce")

    data["sma_5"] = close.rolling(5).mean()
    data["sma_20"] = close.rolling(20).mean()

    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()

    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(
        span=9, adjust=False
    ).mean()
    data["macd_histogram"] = (
        data["macd"] - data["macd_signal"]
    )

    data["momentum"] = close.pct_change(5) * 100

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss
    data["rsi"] = 100 - (100 / (1 + rs))

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["atr"] = true_range.rolling(14).mean()

    data["volume_avg"] = volume.rolling(20).mean()
    data["volume_ratio"] = volume / data["volume_avg"]

    data["high_20"] = high.rolling(20).max()
    data["low_20"] = low.rolling(20).min()

    return data


def get_latest_indicators(history):
    if history is None or history.empty:
        return None

    data = calculate_indicators(history)

    if data.empty:
        return None

    latest = data.iloc[-1]

    result = {}

    for feature in FEATURES:
        value = latest.get(feature)

        if value is None or pd.isna(value):
            return None

        value = float(value)

        if not np.isfinite(value):
            return None

        result[feature] = value

    result["volume"] = float(latest["Volume"])
    result["volume_avg"] = float(latest["volume_avg"])

    return result
