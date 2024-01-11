import sys
sys.path.insert(0, '..')
import os
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List,Dict,Any,Tuple,Optional
import torch

class llama_model():
    def __init__(self,model_name):
        torch.set_default_device("cuda")
        self.model_name=model_name
        print(f"Loading model {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda", trust_remote_code=True)
        self.model.eval()

    def get_answer(self,prompt:str):
        # for final reasoning
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(input_ids,max_new_tokens=10, temperature=0.2,top_k=40,length_penalty=-1)
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            del output
            torch.cuda.empty_cache()
        return response.replace(prompt,'')

    def wrap_prompt(self,query:str,passage:str)->str:
        message = """<s>[INST] <<SYS>>
            You are Summarizer Pro, a specialized assistant for summarizing documents. Focus on extracting
            and summarizing information strictly relevant to the user's query.
            <</SYS>>"""+f"Here is the passage: {passage}. Summarize the key points relevant to my query: {query}. Your responses should be concise summaries without extra words or explanations.[/INST]" 
        return message
    
class openchat():
    """
    https://huggingface.co/docs/transformers/main/en/model_doc/mistral
    Openchat model based on Mistral
    """
    def __init__(self):
        self.model_name="openchat/openchat-3.5-1210"
        self.tokenizer = AutoTokenizer.from_pretrained("openchat/openchat-3.5-1210")
        self.model = AutoModelForCausalLM.from_pretrained("openchat/openchat-3.5-1210").cuda()
        self.model.eval()

    def get_answer(self, message:List[Dict[str,str]]):
        """
        One turn chat response

        Example:    
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you today?"}
                ]
        """
        with torch.no_grad():
            input_text=self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            input_tokens=self.tokenizer(input_text, return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(**input_tokens, max_new_tokens=512,pad_token_id=self.tokenizer.eos_token_id)
            return self.decode_redundant(generated_ids)
    
    def decode_redundant(self,generated_ids):
        return self.tokenizer.decode(generated_ids[0]).split("<|end_of_turn|>")[-2].strip().split(":")[-1].strip()
