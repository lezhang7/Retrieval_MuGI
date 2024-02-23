import sys
sys.path.insert(0, '..')
import os
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List,Dict,Any,Tuple,Optional
import torch

device = "cuda" # the device to load the model onto


class HuggingFaceLanguageModel():  
    def __init__(self,model_name:str):
        self.model_name=model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype='auto'
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    def get_response(self, messages):
        if 'Qwen' in self.model_name:
            return self.chat_qwen(messages)
        elif '01-ai' in self.model_name:
            return self.chat_yi(messages)
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")
    def chat_qwen(self, messages: List[str]) -> str:
        
        text = self.tokenizer.apply_chat_template(
            messages[:-1], # remove the last message containing  "role": "assistant"
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    def chat_yi(self, messages: List[str]) -> str:
        input_ids = self.tokenizer.apply_chat_template(conversation=messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors='pt')
        with torch.no_grad():
            output_ids = self.model.generate(input_ids.to('cuda'))
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response