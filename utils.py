import os
import json
import logging
from typing import List, Dict, Tuple, Union
# from rank_gpt import write_eval_file
import tempfile
from trec_eval import EvalFunction

class PromptTemplate():
    def __init__(self):
        pass
    def template(self, query, document=None, type='summarization'):
        
        if type=='sum':
            return PromptTemplate.get_prefix_prompt_sum(query, document)
        elif type=='sa':
            return PromptTemplate.get_prefix_prompt_short_answer(query, document)
        elif type=='gen':
            return PromptTemplate.get_prefix_prompt_pseudo_doc(query, document)
        elif type=='gen_v2':
            return PromptTemplate.get_prefix_prompt_pseudo_doc_v2(query, document)
        elif type=='gen_v3':
            return PromptTemplate.get_prefix_prompt_pseudo_doc_v3(query, document)
        elif type == 'rj': # relevance judgement
            return PromptTemplate.get_relevance_judgement(query, document)
        else:
            raise NotImplementedError(f"template type {type} is not implemented.")
        # implement other templates here

    @staticmethod
    def get_prefix_prompt_rank(query, documents:List[str]):
        num = len(documents)
        messages = [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'},]
        for idx,document in enumerate(documents):
            messages.append({'role': 'user', 'content': f"[{idx}] {document}"})
            messages.append({'role': 'assistant', 'content': f'Received passage [{idx}].'})
        messages.append({'role': 'user', 'content': f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."})

        return messages
        

    
    @staticmethod
    def get_prefix_prompt_sum(query, document):
        return [{'role': 'system',
                'content': "You are Summarizer Pro, a specialized assistant for summarizing documents. Focus on extracting and summarizing information from a passage. Your responses should be concise summaries without extra words or explanations."},
                {'role': 'assistant', 'content': 'Understood. Please share the document.'},
                {'role': 'user', 'content': f"Here is the document: {document}. Summarize the key points: "}]
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
    def get_prefix_prompt_pseudo_doc_zs(query,document=None): #dl19 ndcg 100: 71.36 @10 1k: 73.6 ndcg@10
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
    @staticmethod
    def get_prefix_prompt_streamline(query,document): #dl19 ndcg 100: 72.12 @10 1k: 74.16 ndcg@10  
        return [
                {
                    "role": "system",
                    "content": "You are StreamlinePro. Plesae remove the redudant sentence that is not relevant to the query. You should only delete without other operations."
                },
                {
                    "role": "user",
                    "content": "For example: Given the query 'how long is the life cycle of a flea'. The redudant document is:: 'The life cycle of a flea typically lasts around 2-3 months, although it can vary depending on environmental conditions. Fleas undergo complete metamorphosis, which consists of four stages: egg, larva, pupa, and adult. The entire life cycle can be completed in as little as 2 weeks under ideal conditions. Flea eggs are laid on the host animal and then fall off into the environment, where they hatch into larvae. The larvae feed on organic matter and develop into pupae, which eventually emerge as adult fleas. Adult fleas then seek a host to feed on and reproduce, starting the cycle anew. It's important to note that proper flea control measures are necessary to prevent infestations and ensure the well-being of both pets and humans.' The streamlined version is: 'The life cycle of a flea typically lasts around 2-3 months, although it can vary depending on environmental conditions. The entire life cycle can be completed in as little as 2 weeks under ideal conditions.'."
                },
                {
                    "role": "user",
                    "content": f"Now, Given the query '{query}', please streamline the document '{document}'"
                },
                {
                    "role": "assistant",
                    "content": "The streamlined version is:"
                }]
    @staticmethod
    def get_prefix_prompt_streamline_v2(query,document): #dl19 ndcg 100:  @10 1k:  ndcg@10  
        return [
                {
                    "role": "user",
                    "content": f"""Plesae remove the redudant sentence that is not relevant to the query. You should only delete without other operations.\nFor example given the query: 'how long is life cycle of flea'.\nThe redudant document is:"The life cycle of a flea typically lasts around 2-3 months, although it can vary depending on environmental conditions. Fleas undergo complete metamorphosis, which consists of four stages: egg, larva, pupa, and adult. The entire life cycle can be completed in as little as 2 weeks under ideal conditions. Flea eggs are laid on the host animal and then fall off into the environment, where they hatch into larvae. The larvae feed on organic matter and develop into pupae, which eventually emerge as adult fleas. Adult fleas then seek a host to feed on and reproduce, starting the cycle anew. It's important to note that proper flea control measures are necessary to prevent infestations and ensure the well-being of both pets and humans."\nThe streamlined version is:"The life cycle of a flea typically lasts around 2-3 months, although it can vary depending on environmental conditions. The entire life cycle can be completed in as little as 2 weeks under ideal conditions."\n\nNow, Given the query: '{query}'please streamline the document:'{document}'"""},
        
                {
                    "role": "assistant",
                    "content": "The streamlined version is:"
                }]



    @staticmethod
    def get_relevance_judgement(query, document):
        TEMPLATE = """
        Query:

        {query}

        Content:

        {content}

        Did the content directly answer the query? If yes, give me the exact sentence(s) without additional content. If no, just answer "irrelevant".
        """
        prompt = TEMPLATE.format(query=query, content=document)
        return  [{'role': 'user', 'content': prompt}]
    

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
