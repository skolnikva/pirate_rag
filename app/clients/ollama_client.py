from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

PIRATE_SYSTEM_PROMPT = (
    "Ты старый морской пират. Отвечай всегда как пират: "
    "используй слова 'Йо-хо-хо', 'шхуна', 'абордаж', 'ром'. "
    "Пиши коротко, ярко и весело. Ответы на русском."
)


def build_messages(messages: list):
    lc_messages = [SystemMessage(content=PIRATE_SYSTEM_PROMPT)]
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def run_ollama(model: str, messages: list, temperature: float):
    lc_messages = build_messages(messages)
    llm = ChatOllama(model=model, temperature=temperature)
    response = llm.invoke(lc_messages)
    return response.content
