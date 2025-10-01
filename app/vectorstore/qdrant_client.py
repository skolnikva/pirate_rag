from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams


def get_qdrant_client(recreate: bool = False):
    client = QdrantClient(host="localhost", port=6333)
    if recreate:
        client.recreate_collection(
            collection_name="diploma_rus",
            vectors_config=VectorParams(size=312, distance="Cosine")
        )
    else:
        pass
    return client
