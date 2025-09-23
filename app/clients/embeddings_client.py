from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
