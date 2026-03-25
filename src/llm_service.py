import os

from ibm_watsonx_ai.foundation_models import Model
from langchain_ibm import WatsonxLLM
from langchain_ollama import OllamaLLM


def get_llm(model_id, parameters, credentials, project_id):
    print(f"Initializing IBM Watsonx LLM: {model_id}...")
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
    )
    llm = WatsonxLLM(model=model)
    return llm


def get_ollama_llm(model_id="llama3.1"):
    print(f"Initializing Local Ollama LLM: {model_id}...")

    # Normalize phi3 shorthand to a pulled tag
    if model_id == "phi3":
        model_id = os.getenv("OLLAMA_MODEL_NORMALIZED", "phi3:mini")

    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "96"))
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

    llm = OllamaLLM(
        model=model_id,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    return llm
