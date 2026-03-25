import os
import pathlib
import sys

from dotenv import load_dotenv

# Ensure repo root is on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document_processor import process_document, create_vectorstore
from src.llm_service import get_ollama_llm
from src.rag_chain import create_conversational_chain

load_dotenv()


def main():
    # Paths and env defaults
    file_path = os.getenv("SMOKE_FILE", "companyPolicies.txt")
    question = os.getenv("SMOKE_QUESTION", "What is the policy?")

    embed_backend = os.getenv("EMBED_BACKEND", "ollama")
    embed_model = os.getenv("EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    llm_model = os.getenv("OLLAMA_MODEL", "phi3:mini")

    print(f"Using file: {file_path}")
    print(f"Question: {question}")
    print(f"Embed backend: {embed_backend} | Embed model: {embed_model}")
    print(f"LLM model: {llm_model}")

    texts = process_document(file_path)
    vectorstore = create_vectorstore(texts, embedding_model=embed_model, prefer=embed_backend)
    llm = get_ollama_llm(llm_model)
    qa_chain = create_conversational_chain(llm, vectorstore)

    result = qa_chain.invoke({"question": question})
    answer = result.get("answer", "No answer found.")
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
