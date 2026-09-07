from fastapi import FastAPI

app = FastAPI(
    title="Stock Assistant",
    description="AI-powered stock research assistant",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Stock Assistant is running!",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
