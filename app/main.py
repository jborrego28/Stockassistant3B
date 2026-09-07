from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.stock_data import get_stock_data


app = FastAPI(
    title="Stock Assistant",
    description="AI-powered stock research assistant",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request, symbol: str | None = None):
    stock = None

    if symbol:
        try:
            stock = get_stock_data(symbol)
        except Exception as e:
            print(f"Stock error: {e}")
            stock = None

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "symbol": symbol or "",
            "stock": stock,
        },
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/stock/{symbol}")
def stock(symbol: str):
    data = get_stock_data(symbol)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to retrieve stock data for {symbol.upper()}",
        )

    return data
