import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List, Dict
from torch import Tensor

task = 'Given a web search query, retrieve relevant passages that answer the query'

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]
    
def mean_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    return torch.sum(last_hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Reranker:
    def __init__(self, model_name, mode):
        self.model_name = model_name 
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mode = mode

    def embed(self, input_texts):
        if "e5-mistral-7b-instruct" in self.model_name:
            if type(input_texts) != list:
                input_texts = [input_texts]
            batch_dict = self.tokenizer(
                input_texts,
                max_length=511,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )
            batch_dict["input_ids"] = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in batch_dict["input_ids"]
            ]
            batch_dict = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
        else:
            input_tokens = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = self.model(**input_tokens)
                if "bge-large-en-v1.5" in self.model_name:
                    embeddings = outputs[0][:, 0]
                else:
                    embeddings = mean_pooling(outputs.last_hidden_state, input_tokens['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def query_enhance(self, item, gen_key):
        query_modes = {
            "alternate":   item["query"]
                + item["gen_key"][0]
                + item["query"]
                + item["gen_key"][1]
                + item["query"]
                + item["gen_key"][2],
            "concat": item['query'] + ''.join(item[gen_key]),
            "query": get_detailed_instruct(task, item['query']) if "mistral" in self.model_name else item['query'],
            "qg": item['query'] + item[gen_key][0]
        }
        return query_modes.get(self.mode, item['query'])

    def rerank(self, rank_result: List[Dict], gen_key: str, topk=100, use_enhanced_query=False):
        rerank_result = {}
        for item in tqdm(rank_result):
            query = self.query_enhance(item, gen_key) if use_enhanced_query else item['query']
            query_embed = self.embed(query)
            docs = [hit['content'] for hit in item['hits'][:topk]]
            docs_idx = [hit['docid'] for hit in item['hits'][:topk]]
            hits_embed = self.embed(docs)
            result = torch.matmul(query_embed, hits_embed.T)
            _, indices = result.topk(10, dim=1)
            selected_doc_ids = [docs_idx[i] for i in indices.squeeze().tolist()]
            rerank_result[item['hits'][0]['qid']] = selected_doc_ids
        return rerank_result
