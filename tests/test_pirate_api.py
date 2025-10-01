import pytest
import pytest_asyncio
from main import app
from app.clients.embeddings_client import get_embeddings_model
from app.vectorstore.qdrant_client import get_qdrant_client
from langchain_qdrant import QdrantVectorStore
from app.services.pirate_service import generate_pirate_reply, strip_think
from openai import AsyncOpenAI
from httpx import AsyncClient, ASGITransport

openai_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1", api_key="dummy"
)


@pytest_asyncio.fixture(scope="session")
async def qdrant_client():
    return get_qdrant_client()


@pytest_asyncio.fixture(scope="session")
async def embeddings_model():
    return get_embeddings_model()


@pytest_asyncio.fixture(scope="function")
async def asgi_client():
    async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_embeddings_and_qdrant(qdrant_client, embeddings_model):
    """Проверка эмбеддингов и поиска в Qdrant"""
    emb = embeddings_model
    vecs = emb.embed_documents(["тест"])
    assert len(vecs[0]) > 0, "Эмбеддинг пустой"

    vs = QdrantVectorStore(
        client=qdrant_client, collection_name="diploma_rus", embedding=emb)

    docs = vs.similarity_search("Пользовательский интерфейс", k=3)
    print("similarity_search вернул документы:", len(docs))
    for i, d in enumerate(docs, 1):
        print(f"--- doc {i} ---")
        print(d.page_content[:200])

    assert isinstance(docs, list)


@pytest.mark.asyncio
async def test_rag_chain():
    """Проверка, что RAG выдаёт осмысленный ответ"""
    messages = [{"role": "user", "content": "Что такое UI Kit?"}]
    result = await generate_pirate_reply(
        messages,
        model="qwen3:4b",
        temperature=0.7
    )
    result = strip_think(result)
    print("RAG result:", result)

    assert isinstance(result, str)
    assert any(w in result.lower()
               for w in ["йо", "шхуна", "абордаж", "ром"]), \
        "Ответ не в пиратском стиле"


@pytest.mark.asyncio
async def test_pirate_reply_with_prompt(asgi_client):
    payload = {
        "model": "qwen3:4b",
        "messages": [{"role": "user", "content": "Какие поля "
                      "должны быть в форме регистрации?"}],
        "temperature": 0.7
    }
    response = await asgi_client.post("/v1/chat/completions", json=payload)
    if response.status_code != 200:
        print("Error response body:", response.text)
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}"

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    assert any(word in content.lower()
               for word in ["йо-хо-хо", "шхуна", "абордаж", "ром"])
    print("Pirate reply:", content)


@pytest.mark.asyncio
async def test_openai_compatibility():
    """Проверка совместимости с OpenAI API"""
    resp = await openai_client.chat.completions.create(
        model="qwen3:4b",
        messages=[{"role": "user", "content": "Что такое UI Kit"}]
    )

    assert hasattr(resp, "id")
    assert hasattr(resp, "object")
    assert hasattr(resp, "choices")

    choice = resp.choices[0]
    assert hasattr(choice, "message")
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)

    print("OpenAI-compatible reply:", choice.message.content)
