import re
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from app.clients.ollama_client import PIRATE_SYSTEM_PROMPT
from app.clients.embeddings_client import get_embeddings_model
from app.vectorstore.qdrant_client import get_qdrant_client


PIRATE_SYSTEM_SAFE = PIRATE_SYSTEM_PROMPT + (
    "\n\nНЕ выводи внутренние рассуждения, не используй <think>...</think>."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(PIRATE_SYSTEM_SAFE),
    HumanMessagePromptTemplate.from_template(
        "Используй только контекст из документов ниже. "
        "Если в документах нет ответа, скажи пиратским стилем: "
        "'Йо-хо-хо! В сундуке пусто, ром закончился!' "
        "\n\nКонтекст:\n{context}\n\nВопрос: {question}"
    ),
])


def strip_think(text: str) -> str:
    """
    Удаляет скрытые размышления модели (<think>...</think>).
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_user_message(messages: list) -> str | None:
    """
    Достаёт последний пользовательский вопрос из истории сообщений.
    """
    return next((m["content"]
                 for m in reversed(messages) if m["role"] == "user"), None)


def get_retriever():
    """
    Создаёт retriever для поиска в Qdrant.
    """
    qdrant_client = get_qdrant_client()
    embeddings = get_embeddings_model()
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name="diploma_rus",
        embedding=embeddings,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


def build_rag_chain(model: str, temperature: float, retriever):
    """
    Создаёт RetrievalQA цепочку.
    """
    llm = ChatOllama(model=model, temperature=temperature)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
        input_key="question",
        )


async def generate_pirate_reply(
        messages: list,
        model: str = "qwen3:4b",
        temperature: float = 0.7
        ) -> str:
    """
    Генерирует пиратский ответ на вопрос пользователя с помощью RAG.
    """
    user_msg = extract_user_message(messages)
    if not user_msg:
        return "Йо-хо-хо! Не вижу вопроса, абордаж!"

    retriever = get_retriever()
    rag_chain = build_rag_chain(model, temperature, retriever)

    output = await rag_chain.ainvoke({"question": user_msg})
    context_docs = output.get("source_documents", [])

    if not context_docs:
        return "Йо-хо-хо! В сундуке пусто, ром закончился!"

    return strip_think(output["result"])
