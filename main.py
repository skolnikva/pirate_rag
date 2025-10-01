from fastapi import FastAPI
from app.routers import pirate_router

app = FastAPI(title="Pirate API (Ollama + LangChain)", docs_url="/docs")

app.include_router(pirate_router.router, prefix="/v1")
