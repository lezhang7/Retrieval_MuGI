from .llm import *
from .reranker import *
HUGGINGFACE_MODELS = ["Qwen/Qwen1.5-72B-Chat-AWQ", "Qwen/Qwen1.5-7B-Chat-AWQ", "Qwen/Qwen1.5-14B-Chat-AWQ","01-ai/Yi-34B-Chat-4bits","01-ai/Yi-6B-Chat-4bits"]
def get_language_model(model_name):
    """
    Load language model instance
        model_name: str, name of the model
    """
    if model_name in HUGGINGFACE_MODELS:
        return HuggingFaceLanguageModel(model_name)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented, supported models are {HUGGINGFACE_MODELS}")

def get_reranker(model_name, mode):
    return Reranker(model_name, mode)