import sys
import os
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.config import CREDENTIALS, PROJECT_ID, DEFAULT_MODEL_ID, DEFAULT_PARAMETERS
from src.document_processor import download_document, process_document, create_vectorstore
from src.llm_service import get_llm, get_ollama_llm
from src.rag_chain import create_conversational_chain

def suppress_warnings():
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    warnings.filterwarnings('ignore')

def main():
    suppress_warnings()
    
    # Setup document (offline dostu)
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
    filename = download_document(url)
    
    # Process and Index
    texts = process_document(filename)
    embed_backend = os.getenv("EMBED_BACKEND", "ollama")
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = create_vectorstore(texts, embedding_model=embed_model, prefer=embed_backend)
    
    # Initialize LLM (IBM varsa onu, yoksa yerel Ollama)
    if CREDENTIALS.get("api_key"):
        llm = get_llm(DEFAULT_MODEL_ID, DEFAULT_PARAMETERS, CREDENTIALS, PROJECT_ID)
    else:
        local_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        print(f"\nNo IBM API key found; switching to local Ollama model '{local_model}'.")
        llm = get_ollama_llm(local_model)
    
    # Create Conversational Chain
    qa_chain = create_conversational_chain(llm, vectorstore)
    
    print("\n--- RAG Chatbot Ready! ---")
    print("Type 'quit', 'exit', or 'bye' to stop.\n")
    
    while True:
        try:
            query = input("Question: ")
            if query.lower() in ["quit", "exit", "bye"]:
                print("Answer: Goodbye!")
                break
            
            if not query.strip():
                continue
                
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "No answer found.")
            
            print(f"Answer: {answer}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
