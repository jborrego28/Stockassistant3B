import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.indicators import FEATURES


def _vector(indicators):
    values = []

    for feature in FEATURES:

        value = indicators.get(feature)

        if value is None or not np.isfinite(value):
            return None

        values.append(float(value))

    return values


def train_model(training_samples):

    X = []
    y = []

    for sample in training_samples:

        indicators = sample.get("indicators")

        if not indicators:
            continue

        vector = _vector(indicators)

        if vector is None:
            continue

        X.append(vector)
        y.append(int(sample["target"]))

    if len(X) < 100 or len(set(y)) < 2:
        return None

    X = np.array(X)
    y = np.array(y)

    # ---------------------------------------------------------
    # VALIDATION DATA
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # VALIDATION ACCURACY
    # ---------------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    model.validation_accuracy = accuracy

    return model


def predict_with_model(
    model,
    indicators,
    current_price,
):

    # ---------------------------------------------------------
    # MODEL UNAVAILABLE
    # ---------------------------------------------------------

    if model is None:

        return {
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
            "neutral_probability": 1.0,
            "reasons": [
                "Model unavailable."
            ],
            "daily_forecast": [],
        }

    # ---------------------------------------------------------
    # CREATE FEATURE VECTOR
    # ---------------------------------------------------------

    vector = _vector(indicators)

    if vector is None:

        return {
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
            "neutral_probability": 1.0,
            "reasons": [
                "Insufficient technical data."
            ],
            "daily_forecast": [],
        }

    # ---------------------------------------------------------
    # PREDICTION PROBABILITIES
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        [vector]
    )[0]

    classes = list(
        model.classes_
    )

    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities,
        )
    }

    # ---------------------------------------------------------
    # CLASS DEFINITIONS
    #
    # 0 = Bearish
    # 1 = Neutral
    # 2 = Bullish
    # ---------------------------------------------------------

    bearish = probability_map.get(
        0,
        0.0,
    )

    neutral = probability_map.get(
        1,
        0.0,
    )

    bullish = probability_map.get(
        2,
        0.0,
    )

    # ---------------------------------------------------------
    # DETERMINE SIGNAL
    # ---------------------------------------------------------

    probability_values = {
        "Bullish": bullish,
        "Neutral": neutral,
        "Bearish": bearish,
    }

    signal = max(
        probability_values,
        key=probability_values.get,
    )

    probability = probability_values[
        signal
    ]

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    probability_confidence = (
        probability * 100
    )

    validation_accuracy = getattr(
        model,
        "validation_accuracy",
        0.50,
    )

    accuracy_confidence = (
        validation_accuracy * 100
    )

    confidence = (
        probability_confidence * 0.65
        +
        accuracy_confidence * 0.35
    )

    confidence = max(
        50,
        min(
            95,
            confidence,
        ),
    )

    confidence = round(
        confidence
    )

    # ---------------------------------------------------------
    # EXPECTED PRICE MOVEMENT
    # ---------------------------------------------------------

    bullish_strength = bullish
    bearish_strength = bearish

    directional_score = (
        bullish_strength
        -
        bearish_strength
    )

    # Maximum expected movement is 8%.
    target_change = (
        directional_score * 0.08
    )

    # Prevent tiny meaningless movements.

    target_price = (
        current_price
        *
        (1 + target_change)
    )

    # ---------------------------------------------------------
    # MODEL SCORE
    # ---------------------------------------------------------

    score = (
        bullish
        -
        bearish
    ) * 100

    # ---------------------------------------------------------
    # MODEL REASONS
    # ---------------------------------------------------------

    reasons = []

    if indicators["sma_5"] > indicators["sma_20"]:

        reasons.append(
            "Short-term trend is above the 20-day trend."
        )

    else:

        reasons.append(
            "Short-term trend is below the 20-day trend."
        )

    if indicators["ema_12"] > indicators["ema_26"]:

        reasons.append(
            "12-day EMA is above the 26-day EMA."
        )

    else:

        reasons.append(
            "12-day EMA is below the 26-day EMA."
        )

    if indicators["macd"] > indicators["macd_signal"]:

        reasons.append(
            "MACD momentum is positive."
        )

    else:

        reasons.append(
            "MACD momentum is negative."
        )

    rsi = indicators["rsi"]

    if 45 <= rsi <= 70:

        reasons.append(
            "RSI is in a relatively healthy range."
        )

    elif rsi > 70:

        reasons.append(
            "RSI is elevated."
        )

    else:

        reasons.append(
            "RSI indicates weaker momentum."
        )

    if indicators["momentum"] > 1:

        reasons.append(
            "Recent momentum is positive."
        )

    elif indicators["momentum"] < -1:

        reasons.append(
            "Recent momentum is negative."
        )

    # ---------------------------------------------------------
    # 5-DAY DAILY FORECAST
    # ---------------------------------------------------------

    daily_forecast = []

    base_price = float(
        current_price
    )

    for day in range(1, 6):

        # Progress toward the final
        # 5-day target.
        day_fraction = day / 5

        daily_change = (
            target_change
            *
            day_fraction
        )

        forecast_price = (
            base_price
            *
            (1 + daily_change)
        )

        forecast_change_percent = (
            daily_change * 100
        )

        dollar_change = (
            forecast_price
            -
            base_price
        )

        # -----------------------------------------------------
        # DAILY SIGNAL
        # -----------------------------------------------------

        if daily_change > 0.002:

            daily_signal = "Bullish"

        elif daily_change < -0.002:

            daily_signal = "Bearish"

        else:

            daily_signal = "Neutral"

        # -----------------------------------------------------
        # DAILY FORECAST ENTRY
        # -----------------------------------------------------

        daily_forecast.append({

            "day": day,

            "price": round(
                forecast_price,
                2,
            ),

            "dollar_change": round(
                dollar_change,
                2,
            ),

            "change_percent": round(
                forecast_change_percent,
                2,
            ),

            "signal": daily_signal,

            "confidence": confidence,

        })

    # ---------------------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------------------

    return {

        "signal": signal,

        "confidence": confidence,

        "target_price": round(
            target_price,
            2,
        ),

        "target_change_percent": round(
            target_change * 100,
            2,
        ),

        "score": round(
            score,
            2,
        ),

        "bullish_score": round(
            bullish * 100,
            2,
        ),

        "bearish_score": round(
            bearish * 100,
            2,
        ),

        "bullish_probability": round(
            bullish,
            4,
        ),

        "bearish_probability": round(
            bearish,
            4,
        ),

        "neutral_probability": round(
            neutral,
            4,
        ),

        "reasons": reasons,

        "daily_forecast": daily_forecast,

    }


def predict_stock(
    indicators,
    current_price,
):

    return {

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

        "neutral_probability": 1.0,

        "reasons": [
            "A trained ML model is required."
        ],

        "daily_forecast": [],

    }
