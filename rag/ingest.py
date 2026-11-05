
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


def ingest_documents():
    documents = []

    # 1. Load PDF documents
    for pdf_file in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    if not documents:
        print("No PDF documents found in data/")
        return

    print(f"Loaded {len(documents)} pages")

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # 3. Generate embeddings and store in Chroma
    vector_store = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    vector_store.add_documents(chunks)

    print("Documents successfully stored in Chroma!")


if __name__ == "__main__":
    ingest_documents()
