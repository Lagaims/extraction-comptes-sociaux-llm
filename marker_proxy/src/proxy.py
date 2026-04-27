from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
import openai as openai_errors
import os
from dotenv import load_dotenv
import logging
from langfuse import get_client
from langfuse.openai import AsyncOpenAI

load_dotenv()

REAL_LLM_BASE_URL = os.getenv("REAL_LLM_BASE_URL", "").rstrip("/")
REAL_LLM_API_KEY = os.getenv("REAL_LLM_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(base_url=REAL_LLM_BASE_URL, api_key=REAL_LLM_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await client.close()
    get_client().flush()


app = FastAPI(
    title="LLM Proxy with Langfuse",
    version="1.0.0",
    description="Proxy for LLM requests with Langfuse tracing",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    try:
        response = await client.chat.completions.create(**data)
        return response.model_dump()
    except openai_errors.APIStatusError as e:
        logger.error(f"LLM error {e.status_code}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)


@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json()
    try:
        response = await client.completions.create(**data)
        return response.model_dump()
    except openai_errors.APIStatusError as e:
        logger.error(f"LLM error {e.status_code}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)


@app.get("/v1/models")
async def list_models():
    try:
        models = await client.models.list()
        return models.model_dump()
    except openai_errors.APIStatusError as e:
        logger.error(f"Models error {e.status_code}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "langfuse_configured": True,
        "real_llm_configured": bool(REAL_LLM_API_KEY),
    }
