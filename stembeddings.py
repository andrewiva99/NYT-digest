from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class STEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path, device='cpu')

    def embed_documents(self, texts: list[str]):
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str):
        return self.model.encode([text])[0].tolist()