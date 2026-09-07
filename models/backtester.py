import yfinance as yf

from models.indicators import get_latest_indicators
from models.predictor import train_model, predict_with_model


def _make_target(current_price, future_price):
    change = (future_price - current_price) / current_price

    # Binary target:
    # 1 = price rises over the next 5 trading days
    # 0 = price falls or stays flat
    return 1 if change > 0 else 0


def _accuracy(predictions, minimum_confidence=0):
    filtered = [
        p for p in predictions
        if p["confidence"] >= minimum_confidence
        and p["signal"] != "Neutral"
    ]

    if not filtered:
        return {
            "predictions": 0,
            "correct": 0,
            "accuracy": 0,
        }

    correct = sum(
        1 for p in filtered if p["correct"]
    )

    return {
        "predictions": len(filtered),
        "correct": correct,
        "accuracy": round(
            correct / len(filtered) * 100,
            2,
        ),
    }


def backtest_stock(symbol, period="1y", forward_days=5):
    symbol = symbol.upper().strip()

    history = yf.Ticker(symbol).history(period=period)

    if history.empty:
        return None

    # Enough history for indicators + meaningful training.
    minimum_history = 120

    if len(history) <= minimum_history + forward_days:
        return None

    predictions = []

    bullish_calls = 0
    bearish_calls = 0
    neutral_calls = 0

    bullish_correct = 0
    bearish_correct = 0

    returns = []

    model = None
    last_train = -999
    retrain_every = 20

    for i in range(
        minimum_history,
        len(history) - forward_days,
    ):

        # --------------------------------------------------
        # WALK-FORWARD TRAINING
        # --------------------------------------------------
        #
        # At prediction date i, ONLY information available
        # before i is allowed in the training set.
        #
        if model is None or i - last_train >= retrain_every:

            training_samples = []

            # j + forward_days must be strictly before i.
            training_end = i - forward_days

            for j in range(30, training_end):

                historical_data = history.iloc[:j + 1].copy()

                indicators = get_latest_indicators(
                    historical_data
                )

                if indicators is None:
                    continue

                current_price = float(
                    historical_data["Close"].iloc[-1]
                )

                future_price = float(
                    history["Close"].iloc[
                        j + forward_days
                    ]
                )

                target = _make_target(
                    current_price,
                    future_price,
                )

                training_samples.append({
                    "indicators": indicators,
                    "target": target,
                })

            model = train_model(training_samples)

            if model is None:
                continue

            last_train = i

        # --------------------------------------------------
        # PREDICTION DATA
        # --------------------------------------------------

        historical_data = history.iloc[:i + 1].copy()

        indicators = get_latest_indicators(
            historical_data
        )

        if indicators is None:
            continue

        current_price = float(
            historical_data["Close"].iloc[-1]
        )

        prediction = predict_with_model(
            model,
            indicators,
            current_price,
        )

        signal = prediction["signal"]
        confidence = prediction["confidence"]

        # --------------------------------------------------
        # FUTURE RESULT
        # --------------------------------------------------

        future_price = float(
            history["Close"].iloc[
                i + forward_days
            ]
        )

        actual_change = (
            future_price - current_price
        ) / current_price * 100

        returns.append(actual_change)

        correct = False

        if signal == "Bullish":

            bullish_calls += 1

            if actual_change > 0:
                bullish_correct += 1
                correct = True

        elif signal == "Bearish":

            bearish_calls += 1

            if actual_change <= 0:
                bearish_correct += 1
                correct = True

        else:
            neutral_calls += 1

        predictions.append({
            "date": history.index[i].strftime("%Y-%m-%d"),
            "signal": signal,
            "confidence": confidence,
            "current_price": round(current_price, 2),
            "future_price": round(future_price, 2),
            "actual_change_percent": round(actual_change, 2),
            "correct": correct,
        })

    directional_predictions = (
        bullish_calls + bearish_calls
    )

    correct_predictions = (
        bullish_correct + bearish_correct
    )

    accuracy = (
        correct_predictions /
        directional_predictions *
        100
        if directional_predictions
        else 0
    )

    bullish_accuracy = (
        bullish_correct /
        bullish_calls *
        100
        if bullish_calls
        else 0
    )

    bearish_accuracy = (
        bearish_correct /
        bearish_calls *
        100
        if bearish_calls
        else 0
    )

    average_return = (
        sum(returns) / len(returns)
        if returns
        else 0
    )

    confidence_accuracy = {}

    for threshold in (
        50, 55, 60, 65, 70,
        75, 80, 85, 90,
    ):
        confidence_accuracy[str(threshold)] = _accuracy(
            predictions,
            threshold,
        )

    return {
        "symbol": symbol,
        "period": period,
        "forward_days": forward_days,
        "total_predictions": len(predictions),
        "directional_predictions": directional_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": round(accuracy, 2),
        "meaningful_accuracy": round(accuracy, 2),
        "bullish_calls": bullish_calls,
        "bullish_correct": bullish_correct,
        "bullish_accuracy": round(bullish_accuracy, 2),
        "bearish_calls": bearish_calls,
        "bearish_correct": bearish_correct,
        "bearish_accuracy": round(bearish_accuracy, 2),
        "neutral_calls": neutral_calls,
        "average_return": round(average_return, 2),
        "confidence_accuracy": confidence_accuracy,
        "predictions": predictions,
    }
