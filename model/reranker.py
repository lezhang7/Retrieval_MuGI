import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List, Dict

task = 'Given a web search query, retrieve relevant passages that answer the query'

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def mean_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    return torch.sum(last_hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Reranker:
    def __init__(self, model_name, mode):
        self.model_name = model_name 
        self.model = AutoModel.from_pretrained(model_name).to('cuda')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mode = mode

    def embed(self, input_texts):
        input_tokens = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(**input_tokens)
            embeddings = mean_pooling(outputs.last_hidden_state, input_tokens['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def query_enhance(self, item, gen_key):
        query_modes = {
            "alternate": item['query'] + ''.join(item[gen_key]),
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
