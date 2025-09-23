from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from vectorstore.qdrant_client import get_qdrant_client
from clients.embeddings_client import get_embeddings_model
from langchain_qdrant import QdrantVectorStore

loader = PyPDFLoader("data/diploma.pdf")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

qdrant_client = get_qdrant_client(recreate=True)
embeddings = get_embeddings_model()
vectorstore = QdrantVectorStore(client=qdrant_client,
                                collection_name="diploma_rus",
                                embedding=embeddings)
vectorstore.add_documents(chunks)

print("PDF загружен в Qdrant!")
