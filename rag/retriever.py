from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

CHROMA_DIR = "chroma_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# 1. Dense retriever
dense_retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# 2. Load existing documents from Chroma
stored = vector_store.get()

from langchain_core.documents import Document

documents = [
    Document(
        page_content=content,
        metadata=metadata or {}
    )
    for content, metadata in zip(
        stored["documents"],
        stored["metadatas"]
    )
]

# 3. Sparse retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=5
)

# 4. Combine both retrievers
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)


def retrieve_documents(question: str):
    return retriever.invoke(question)