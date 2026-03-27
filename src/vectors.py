
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def create_embeddings(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # PDF loading
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No content extracted from PDF.")

        # Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("Failed to create text chunks.")

        # Qdrant connectivity check
        try:
            client = QdrantClient(url=self.qdrant_url)
            collections = client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                client.delete_collection(self.collection_name)
        except Exception as e:
            raise ConnectionError(f"Unable to communicate with Qdrant: {e}")

        # Vector DB creation
        try:
            Qdrant.from_documents(
                splits,
                self.embeddings,
                url=self.qdrant_url,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
        except Exception as e:
            raise RuntimeError(f"Embedding storage in Qdrant failed: {e}")

        return "Vector DB created and stored successfully in Qdrant."
