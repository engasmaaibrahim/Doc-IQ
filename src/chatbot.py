
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import traceback

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        try:
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs=self.encode_kwargs,
            )

            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=self.llm_temperature,
            )

            self.prompt_template = """Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say you don't know.

            Context: {context}
            Question: {question}

            Helpful answer:"""

            self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

            self.db = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=self.collection_name
            )

            self.prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=['context', 'question']
            )

            self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": self.prompt},
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Chatbot initialization failed: {e}")

    def get_response(self, query: str) -> str:
        try:
            return self.qa.run(query)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.text(traceback.format_exc())  
            return "Sorry, I couldn't process your request."
