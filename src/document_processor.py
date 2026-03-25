import os
import time
from functools import lru_cache

import wget
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import (
    FakeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def download_document(url, filename='companyPolicies.txt'):
    if not os.path.exists(filename):
        try:
            print(f"Downloading {filename}...")
            wget.download(url, out=filename)
            print("\nDownload complete.")
        except Exception as exc:
            # Offline senaryosu: varsa mevcut dosyayı kullan
            if os.path.exists(filename):
                print(f"Download failed ({exc}); using existing local file.")
            else:
                raise RuntimeError(f"Download failed and no local file present: {exc}")
    return filename

def process_document(file_path, chunk_size=1500, chunk_overlap=0):
    start = time.perf_counter()
    print(f"Processing {file_path}...")

    # Select loader based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        # PyMuPDF (fitz) daha iyi unicode desteğine sahip; yoksa PyPDFLoader'a düş
        try:
            loader = PyMuPDFLoader(file_path)
            print("PDF loader: PyMuPDFLoader")
        except Exception as exc:
            print(f"PyMuPDFLoader failed ({exc}); falling back to PyPDFLoader.")
            loader = PyPDFLoader(file_path)
    elif ext in ['.xlsx', '.xls']:
        loader = UnstructuredExcelLoader(file_path)
    else:
        loader = TextLoader(file_path, autodetect_encoding=True, encoding="utf-8")
        
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    elapsed = time.perf_counter() - start
    print(f"Split into {len(texts)} chunks in {elapsed:.2f}s.")
    return texts

@lru_cache(maxsize=1)
def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefer: str | None = "ollama",
):
    """
    Offline-öncelikli embedding seçimi.
    Sıra: HuggingFace (yalnızca lokal cache) -> Ollama -> FakeEmbeddings.
    """
    prefer = prefer or "ollama"

    if prefer == "hf":
        try:
            print(f"Loading HF embeddings: {model_name} ...")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder="models",
            )
        except Exception as exc:
            print(f"HF embeddings unavailable ({exc}). Trying Ollama...")

    try:
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        print(f"Loading Ollama embeddings: {embed_model} ...")
        return OllamaEmbeddings(model=embed_model)
    except Exception as exc:
        print(f"Ollama embeddings unavailable ({exc}). Falling back to FakeEmbeddings.")
        return FakeEmbeddings(size=768)


def create_vectorstore(
    texts,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str | None = None,
    prefer: str | None = None,
):
    start = time.perf_counter()
    print("Creating vectorstore (this may take a moment)...")
    embeddings = get_embeddings(embedding_model, prefer=prefer)

    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        db_path = os.path.join(persist_directory, "chroma.sqlite3")
        if os.path.exists(db_path):
            print("Found existing index, loading from disk.")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
            elapsed = time.perf_counter() - start
            print(f"Vectorstore loaded in {elapsed:.2f}s.")
            return vectorstore

    vectorstore = Chroma.from_documents(
        texts, embeddings, persist_directory=persist_directory
    )
    elapsed = time.perf_counter() - start
    print(f"Vectorstore created in {elapsed:.2f}s.")
    return vectorstore
