from fastapi import APIRouter, HTTPException
from app.models.input_text import ChatCompletionRequest
from app.services.pirate_service import generate_pirate_reply

router = APIRouter()


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        reply = await generate_pirate_reply(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
        )
        return {
            "id": "chatcmpl-local-ollama",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop"
                }
            ],
            "model": request.model,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
