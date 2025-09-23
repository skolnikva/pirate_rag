from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        ..., description="qwen3:4b")
    messages: List[Dict[str, Any]] = Field(
        ...,
        description="Список сообщений в формате "
                    "[{'role': 'user', 'content': 'Привет'}]"
    )
    temperature: float = Field(0.7, description="Креативность модели (0-1)")

    model_config = ConfigDict(extra="allow")
