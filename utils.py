import os
import json
import logging
from typing import List, Dict, Tuple, Union
# from rank_gpt import write_eval_file
from pyserini.search import LuceneSearcher, get_topics, get_qrels
import tempfile
from trec_eval import EvalFunction
from openai import OpenAI
import benchmark
class PromptTemplate():
    def __init__(self):
        pass
    def template(self, query, document=None, type='summarization'):
        
        if type=='gen':
            return PromptTemplate.get_prefix_prompt_pseudo_doc_zs(query, document)
        else:
            raise NotImplementedError(f"template type {type} is not implemented.")
        # implement other templates here
        
    @staticmethod
    def get_prefix_prompt_short_answer(query, document=None):
        return [{'role': 'user', 'content': f"{query}, one sentence answering the question: "}]
    @staticmethod
    def get_prefix_prompt_pseudo_doc(query, document=None): # dl19 69.56 ndcg@10
        return [{'role': 'user', 'content': f"'{query}', please write a clear, informative and clear document for answering the query: "}]

    @staticmethod
    def get_prefix_prompt_pseudo_doc_fs(query, document=None): #dl19 100: 70.7 ndcg@10 1k: 74.4 ndcg@10
        return [{'role': 'user', 'content': f"""\nquery: how long is life cycle of flea?\ndocument: The life cycle of a flea typically lasts around 2-3 months, although it can vary depending on environmental conditions. Fleas undergo complete metamorphosis, which consists of four stages: egg, larva, pupa, and adult. The entire life cycle can be completed in as little as 2 weeks under ideal conditions. Flea eggs are laid on the host animal and then fall off into the environment, where they hatch into larvae. The larvae feed on organic matter and develop into pupae, which eventually emerge as adult fleas. Adult fleas then seek a host to feed on and reproduce, starting the cycle anew. It's important to note that proper flea control measures are necessary to prevent infestations and ensure the well-being of both pets and humans.\nquery: cost of interior concrete flooring?\ndocument: "The cost of interior concrete flooring can vary depending on several factors. On average, the cost can range from $2 to $12 per square foot. Factors that can influence the cost include the complexity of the design, the type of concrete finish desired, and any additional treatments or coatings. Basic concrete flooring tends to be more affordable, while decorative options like stamped or stained concrete can be more expensive. It's important to consider the long-term benefits of concrete flooring, such as its durability and low maintenance requirements, when evaluating the overall cost. Additionally, consulting with a professional contractor can provide a more accurate estimate based on your specific project requirements.\nquery: {query}\ndocument: """}]
    @staticmethod
    def get_prefix_prompt_pseudo_doc_zs(query, document = None): #dl19 ndcg 100: 71.36 @10 1k: 73.6 ndcg@10
        return [{
                    "role": "system",
                    "content": "You are PassageGenGPT, an AI capable of generating concise, informative, and clear pseudo passages on specific topics."
                },
                {
                    "role": "user",
                    "content": f"Generate one passage that is relevant to the following query: '{query}'. The passage should be concise, informative, and clear"
                },
                {
                    "role": "assistant",
                    "content": "Sure, here's a passage relevant to the query:"
                }]

def evalute_dict(rank_dict:Dict[str,List[str]],the_topic:str): 
    """
    evaluate the rank_dict, one example is 
    rank_dict = {264014: ['4834547', '6641238', '96855', '3338827', '96851']}
    """
    # Evaluate nDCG@10
    
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_file, 'w') as f:
        for query_id,doc_ids_list in rank_dict.items():
            rank = 1    
            for doc_id in doc_ids_list:
                f.write(f"{query_id} Q0 {doc_id} {rank} {15-rank*0.1} rank\n")
                rank += 1
    return EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', the_topic, temp_file])

def evaluate_bm25(rank_results,the_topic):
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(rank_results, temp_file)
    bm25_rank_score=EvalFunction.eval(['-c', '-m', 'map', the_topic, temp_file])
    return bm25_rank_score

# json load function
def load_json(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'r') as f:
        return json.load(f)

# json dump function
def dump_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def display_args(args):
    print("Program Arguments:")
    max_len = max(len(arg) for arg in vars(args))  # 找到最长的键名长度
    for arg, value in vars(args).items():
        print(f"  {arg.ljust(max_len)}: {value}")

def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

def evaluation_output_files(args):
    evaluation_save_path = os.path.join('results', f"{args.irmode}.json")
    evaluation_results = {}
    if os.path.exists(evaluation_save_path):
        logging.info(f"Loading evaluation results from {evaluation_save_path}")
        evaluation_results = load_json(evaluation_save_path)
    return evaluation_results,evaluation_save_path

def initialize_client():
    global client
    openai_key = os.environ.get('OPENAI_KEY')
    client = OpenAI(api_key=openai_key)


def get_data_pyserini(data,test=False):
    searcher = LuceneSearcher.from_prebuilt_index(benchmark.THE_INDEX[data])
    topics = get_topics(benchmark.THE_TOPICS[data] if data != 'dl20' else 'dl20')
    qrels = get_qrels(benchmark.THE_TOPICS[data])
    topics = {k: v for k, v in topics.items() if k in qrels}
    if test:
        topics = {key:topics[key] for key in list(topics)[:2]}
    return searcher, topics, qrels