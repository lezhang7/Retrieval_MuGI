"""
Given one query embedding, and a list of reference embeddings.
Rank the reference embeddings according to their similairy to the query
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Tuple, Optional
import torch
from torch import Tensor

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    # pooling strategy for e5-base-v2, ember-v1
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def mean_pooling(model_output, attention_mask):
    # pooling strategy for sentence-transformers/all-mpnet-base-v2
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

task = 'Given a web search query, retrieve relevant passages that answer the query'
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class Reranker():
    def __init__(self, model_name, mode):
        self.model_name = model_name 
        self.model =  AutoModel.from_pretrained(model_name).to('cuda')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mode = mode

    def embed(self,input_texts):
        """Implement embedding function here"""

        if "all-mpnet-base-v2" in self.model_name:
            input_tokens = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = self.model(**input_tokens)
                embeddings = mean_pooling(outputs, input_tokens['attention_mask'])
                embeddings = F.normalize(embeddings, dim=-1)
            return embeddings

        elif "e5-base-v2" in self.model_name:
            batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {key: val.to('cuda') for key, val in batch_dict.items()}
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

        elif "ember-v1" in self.model_name:
            #llmrails/ember-v1
            batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        elif "ance" in self.model_name: # model_name = "castorini/ance-msmarco-passage"
            input_tokens = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = self.model(**input_tokens)
                embeddings = mean_pooling(outputs, input_tokens['attention_mask'])
                embeddings = F.normalize(embeddings, dim=-1)
            return embeddings
        else:
            raise NotImplementedError("Please implement your embedding function here")


    def query_enhance(self,item):
        if self.mode == "alternate":
            query = item['query']+item['gen_cand_gpt35'][0]+item['query']+item['gen_cand_gpt35'][1]+item['query']+item['gen_cand_gpt35'][2]
        elif self.mode == "concat":
            query = item['query']+item['gen_cand_gpt35'][0]+item['gen_cand_gpt35'][1]+item['gen_cand_gpt35'][2]
        elif self.mode == "query":
            if "mistral" in self.model_name:
                query = [get_detailed_instruct(task, item['query'])]
            else:
                query = item['query']
        elif self.mode == "qg":
            query = item['query']+item['gen_cand_gpt35'][0]
            # query = item['query']+item['gen_cand_gpt4'][0]
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
        return query
        
    def rerank(self, rank_result:List[Dict],topk=100):
        rerank_result = {}
        for item in tqdm(rank_result):
            query = self.query_enhance(item)
            try:
                query_id = item['hits'][0]['qid']
            except:
                breakpoint()
            docs = [hit['content'] for hit in item['hits'][:topk]]
            docs_idx = [hit['docid'] for hit in item['hits'][:topk]]
            query_embed = self.embed(query) # 1,d
            hits_embed = self.embed(docs)  # 100,d
            result = torch.einsum('bd,cd->bc', query_embed, hits_embed)
            _, indices = torch.sort(result, dim=1, descending=True)
            indices = indices.cpu()[0][:10]
            selected_doc_ids = [docs_idx[i] for i in indices]
            rerank_result[query_id] = selected_doc_ids
        return  rerank_result
    

    

    
    
        

