import yfinance as yf


def get_stock_data(symbol):
    ticker = yf.Ticker(symbol)

    history = ticker.history(period="1mo")

    if history.empty:
        return None

    latest = history.iloc[-1]

    chart = []

    for date, row in history.iterrows():
        chart.append({
            "date": date.strftime("%Y-%m-%d"),
            "price": round(float(row["Close"]), 2),
        })

    return {
        "symbol": symbol.upper(),
        "price": round(float(latest["Close"]), 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "chart": chart,
    }
