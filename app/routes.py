from flask import Blueprint, render_template, request

from app.stock_data import get_stock_data


main = Blueprint("main", __name__)


@main.route("/")
def index():
    symbol = request.args.get("symbol", "AAPL").upper()

    stock = get_stock_data(symbol)

    return render_template(
        "index.html",
        stock=stock,
        symbol=symbol,
    )
