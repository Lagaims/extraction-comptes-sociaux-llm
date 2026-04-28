from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
import openai as openai_errors
import json
import os
from dotenv import load_dotenv
import logging

load_dotenv()

os.environ.setdefault("REAL_LLM_BASE_URL", "https://llm.lab.sspcloud.fr/v1")

REAL_LLM_BASE_URL = os.getenv("REAL_LLM_BASE_URL", "").rstrip("/")
REAL_LLM_API_KEY = os.getenv("REAL_LLM_API_KEY")

LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

if LANGFUSE_ENABLED:
    from langfuse import get_client
    from langfuse.openai import AsyncOpenAI
else:
    from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not LANGFUSE_ENABLED:
    logger.warning("Langfuse credentials not set — tracing disabled, using standard OpenAI client.")

client = AsyncOpenAI(
    base_url=REAL_LLM_BASE_URL,
    api_key=REAL_LLM_API_KEY,
    timeout=None,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await client.close()
    if LANGFUSE_ENABLED:
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
    model = data.get("model", "unknown")
    logger.info(f"→ /v1/chat/completions  model={model}")

    # json_schema (structured outputs) non supporté par tous les LLMs :
    # on convertit en json_object et on injecte le schéma dans le prompt
    if isinstance(data.get("response_format"), dict) and data["response_format"].get("type") == "json_schema":
        schema = data["response_format"].get("json_schema", {}).get("schema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties.keys()))
        if required:
            fields = ", ".join(f'"{k}"' for k in required)
            example = "{" + ", ".join(f'"{k}": ...' for k in required) + "}"
            schema_instruction = (
                f"\n\nYour response MUST be a JSON object with exactly these fields: {fields}.\n"
                f"Format: {example}\n"
                "Return ONLY the JSON object, no explanation."
            )
            messages = data.get("messages", [])
            if messages:
                last = messages[-1]
                if isinstance(last.get("content"), str):
                    last["content"] += schema_instruction
                elif isinstance(last.get("content"), list):
                    last["content"].append({"type": "text", "text": schema_instruction})
        data.pop("response_format", None)

    try:
        response = await client.chat.completions.create(**data)
        logger.info(f"← /v1/chat/completions  model={model}  OK")
        return response.model_dump()
    except openai_errors.APIStatusError as e:
        logger.error(f"LLM error {e.status_code}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except openai_errors.APITimeoutError as e:
        logger.error(f"LLM timeout: {e}")
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except openai_errors.APIConnectionError as e:
        logger.error(f"LLM connection error: {e}")
        raise HTTPException(status_code=502, detail=f"Cannot reach LLM: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json()
    try:
        response = await client.completions.create(**data)
        return response.model_dump()
    except openai_errors.APIStatusError as e:
        logger.error(f"LLM error {e.status_code}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    try:
        models = await client.models.list()
        return models.model_dump()
    except openai_errors.APIStatusError as e:
        logger.error(f"Models error {e.status_code}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "langfuse_configured": LANGFUSE_ENABLED,
        "real_llm_configured": bool(REAL_LLM_API_KEY),
    }
