import yfinance as yf
from datetime import datetime

from models.indicators import get_latest_indicators
from models.predictor import train_model, predict_with_model


def _build_training_samples(history, forward_days=5):
    samples = []

    # Leave enough history for all indicators.
    start = 60
    end = len(history) - forward_days

    for i in range(start, end):
        historical_data = history.iloc[:i + 1].copy()

        indicators = get_latest_indicators(
            historical_data
        )

        if indicators is None:
            continue

        current_price = float(
            historical_data["Close"].iloc[-1]
        )

        future_price = float(
            history["Close"].iloc[i + forward_days]
        )

        change = (
            future_price - current_price
        ) / current_price

        # Same target definition used by the
        # walk-forward backtester.
        if change >= 0.01:
            target = 2
        elif change <= -0.01:
            target = 0
        else:
            target = 1

        samples.append({
            "indicators": indicators,
            "target": target,
        })

    return samples


def get_stock_data(symbol):
    symbol = symbol.upper().strip()

    ticker = yf.Ticker(symbol)

    # ---------------------------------------------------------
    # HISTORICAL DATA
    # ---------------------------------------------------------

    # Keep 2 years of DAILY data for ML training
    # and technical indicators.
    history = ticker.history(
        period="2y",
        interval="1d",
    )

    if history.empty:
        return None

    # ---------------------------------------------------------
    # LIVE / INTRADAY DATA
    # ---------------------------------------------------------

    # Get today's 1-minute data so the displayed price
    # can reflect the latest available market price.
    intraday = ticker.history(
        period="1d",
        interval="1m",
    )

    # Use the most recent intraday price when available.
    # If intraday data isn't available (for example,
    # when the market is closed), fall back to daily data.
    if not intraday.empty:
        latest = intraday.iloc[-1]
    else:
        latest = history.iloc[-1]

    current_price = float(
        latest["Close"]
    )

    # ---------------------------------------------------------
    # DAILY CHANGE
    # ---------------------------------------------------------

    # Use today's opening price when available.
    # This makes the change represent today's movement
    # instead of the 2-year movement.
    if not intraday.empty:
        first_intraday = intraday.iloc[0]

        opening_price = float(
            first_intraday["Open"]
        )

        change = (
            current_price - opening_price
        )

        change_percent = (
            change / opening_price
        ) * 100

    else:
        # Market closed / intraday unavailable.
        # Use the latest daily candle.
        daily_latest = history.iloc[-1]

        opening_price = float(
            daily_latest["Open"]
        )

        change = (
            current_price - opening_price
        )

        change_percent = (
            change / opening_price
        ) * 100

    # ---------------------------------------------------------
    # INDICATORS
    # ---------------------------------------------------------

    indicators = get_latest_indicators(
        history
    )

    if indicators is None:
        return None

    # ---------------------------------------------------------
    # TRAIN LIVE MODEL
    # ---------------------------------------------------------

    training_samples = _build_training_samples(
        history,
        forward_days=5,
    )

    model = train_model(
        training_samples
    )

    if model is None:

        prediction = {
            "signal": "Neutral",
            "confidence": 50,
            "target_price": round(
                current_price,
                2,
            ),
            "target_change_percent": 0,
            "score": 0,
            "bullish_score": 0,
            "bearish_score": 0,
            "reasons": [
                "Not enough historical training data."
            ],
        }

    else:

        prediction = predict_with_model(
            model,
            indicators,
            current_price,
        )

    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------

    # Keep the chart reasonably sized even though
    # the model uses 2 years of history.
    chart_history = history.tail(120)

    chart = []

    for date, row in chart_history.iterrows():

        chart.append({
            "date": date.strftime(
                "%Y-%m-%d"
            ),
            "price": round(
                float(row["Close"]),
                2,
            ),
            "volume": int(
                row["Volume"]
            ),
        })

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    return {
        "symbol": symbol,

        "last_updated": datetime.now().strftime(
            "%b %d, %Y at %I:%M %p"
        ),

        "price": round(
            current_price,
            2,
        ),

        "open": round(
            opening_price,
            2,
        ),

        "high": round(
            float(latest["High"]),
            2,
        ),

        "low": round(
            float(latest["Low"]),
            2,
        ),

        "change": round(
            change,
            2,
        ),

        "change_percent": round(
            change_percent,
            2,
        ),

        "chart": chart,

        "indicators": indicators,

        "prediction": prediction,
    }
