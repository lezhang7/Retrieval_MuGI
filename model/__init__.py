from .llm import *
from .reranker import *
def get_language_model(model_name):
    """
    Load language model instance
        model_name: str, name of the model
    """
    if model_name == "openchat":
        model = openchat()
        return model
    elif model_name == "llama2":
        model=llama_model("meta-llama/Llama-2-13b-chat-hf")
        return model
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

def get_reranker(model_name, mode):
    return Reranker(model_name, mode)