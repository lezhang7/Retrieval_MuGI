from tqdm import tqdm

import os
import json
import utils
from typing import Dict, List
import argparse
import logging
from model import get_language_model, get_reranker
import benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
language_model = None

def ensure_model_loaded(model_name):
    global language_model  
    if language_model is None:  
        language_model = get_language_model(model_name)  
    else:
        print("Model already loaded.") 

def run_gpt(message, model_name='gpt-3.5-turbo'):
    response = client.chat.completions.create(model=model_name,messages=message)
    return response.choices[0].message.content

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--irmode', type=str, default='rerank', choices= ['mugisparse','rerank','mugirerank','mugipipeline'] ,help='information retrieval mode')
    # documents generation setting
    parser.add_argument('--llm', type=str, default='openai', help='pseudo reference generation model')
    parser.add_argument('--doc_gen', type=int, default=5, help='number of generated documents')
    parser.add_argument('--output_path', type=str, default='./exp', help='output path')
    # sparse retrieval setting
    parser.add_argument('--repeat_times', '-t', default=None, type=int)
    parser.add_argument('--adaptive_times', '-at', default=6,type=int)
    parser.add_argument('--topk', type=int, default=100, help='BM25 retrieved topk documents')
    parser.add_argument('--article_num','-a', default=5, type=int)
    # dense retrieval setting
    parser.add_argument('--rank_model', type=str, default='sentence-transformers/all-mpnet-base-v2', help='Text emebdding model ')
    parser.add_argument('--dense_topk', type=int, default=100, help='dense retrieved topk documents')
    parser.add_argument('--mode', type=str, choices=['query', 'alternate', 'concat','qg'], default='concat', help='whether to use generated reference')
    parser.add_argument('--test', action='store_true', help='for fast code test purpose')
    
    args = parser.parse_args()
    return args

def generate_pseudo_references(language_model_name: str, topics: Dict[str, Dict[str, str]], generated_document_num: int):
    """
    Generate pseudo references for the given data in the form of a list of item pairs. Save to output_path.
    Example:
        topics = {query_id: {"title": query_text}}
    Output:
        topics = {query_id: {"title": query_text, "gen_cand_gpt4": [<gen_doc1>, <gen_doc2>, ...], "gen_cand_gpt35": [<gen_doc1>, <gen_doc2>, ...]}}
    """
    for key in tqdm(topics):
        query = topics[key]['title']
        message = utils.PromptTemplate.get_prefix_prompt_pseudo_doc_zs(query)
        
        if 'gpt' in language_model_name:
            global client
            utils.initialize_client()
            models = [
                ('gpt-4-1106-preview', generated_document_num),
                ('gpt-3.5-turbo-1106', min(3, generated_document_num))
            ]
            
            for model_name, doc_num in models:
                gpt_key = 'gpt4' if 'gpt-4' in model_name else 'gpt35'
                gen_key = f'gen_cand_{gpt_key}'
                topics[key].setdefault(gen_key, [])
                for _ in range(doc_num):
                    output = run_gpt(message, model_name=language_model_name).strip()
                    topics[key][gen_key].append(output)
        else:
            gen_key = f'gen_cand_{language_model_name}'
            topics[key].setdefault(gen_key, [])
            for _ in range(generated_document_num):
                output = language_model.get_response(message)
                topics[key][gen_key].append(output)
    return topics, f'gen_cand_4' if 'gpt' in language_model_name else gen_key

def bm25_mugi(args, topics, searcher, qrels, gen_key):
    """
    Perform bm25 with generated pseudo references by gpt.
    Concate the query and pseudo reference in the form of <q>*t+<ref>
    """
    for key in topics:
        query = topics[key]['title']
        gen_ref = ' '.join(topics[key][gen_key][:args.article_num])
        if args.repeat_times:
            times = args.repeat_times
        elif args.adaptive_times:
            times = (len(gen_ref)//len(query))//args.adaptive_times
        topics[key]['enhanced_query'] = (query + ' ')*times + gen_ref
    rank_results = run_retriever(topics, searcher, gen_key, qrels, k=args.topk, use_enhanced_query=True)

    return rank_results

def run_retriever(topics, searcher, gen_key, qrels=None, k=100, qid=None, use_enhanced_query=False):
    """
    Run retriever on a list of topics. If qrels is provided, only run on topics that are in qrels.
    Optionally use an enhanced query if specified.

    Parameters:
    - topics: List of topics or a single topic as a string.
    - searcher: Searcher object to execute search queries.
    - qrels: Optional dictionary of query relevance judgments.
    - k: Number of documents to retrieve.
    - qid: Optional query ID, used when topics is a single string.
    - use_enhanced_query: Flag to use 'enhanced_query' instead of 'title' for topics if available.

    Returns:
    - List of ranked results.
    """
    ranks = []
    if isinstance(topics, str):
        topics = {qid: {'title': topics}} if qid else {'single_query': {'title': topics}}

    for index, (qid, topic) in enumerate(topics.items()):
        if qrels is None or qid in qrels:
            query = topic['enhanced_query'] if use_enhanced_query and 'enhanced_query' in topic else topic['title']
            if index == 0:
                logging.info(f'Running BM25 on query: {query}')
            hits = searcher.search(query, k=k)
            rank_details = {'query': topic['title'], 'hits': []}

            if 'gen_cand_gpt4' in topic:
                rank_details.update({'gen_cand_gpt4': topic['gen_cand_gpt4'], 'gen_cand_gpt35': topic['gen_cand_gpt35']})
            if gen_key in topic:
                rank_details.update({gen_key: topic[gen_key]})

            for rank, hit in enumerate(hits, start=1):
                content = json.loads(searcher.doc(hit.docid).raw())
                formatted_content = 'Title: {} Content: {}'.format(content.get('title', ''), content.get('text', content.get('contents', ''))) if 'title' in content else content.get('contents', '')
                formatted_content = ' '.join(formatted_content.split())
                rank_details['hits'].append({
                    'content': formatted_content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score
                })
            
            ranks.append(rank_details)
    return ranks if len(topics) > 1 else ranks[0]


def get_sparasemugi_result(data, test=False):
    bm25_refine_output_path = os.path.join(args.output_path, args.llm ,data+'_bm25_refine.json')
    if not os.path.exists(bm25_refine_output_path):
        logging.info(f"No local results found for {data}, generating psuedo references and retrieve passages using BM25, saving to {bm25_refine_output_path}.")
        ensure_model_loaded(args.llm)
        try:
            searcher, topics, qrels = utils.get_data_pyserini(data, test)
            logging.info(f'Generating pseudo references for {data}')
            refined_topics, gen_key = generate_pseudo_references(args.llm, topics, args.doc_gen)
            bm25_rank_results = bm25_mugi(args, refined_topics, searcher, qrels, gen_key)
            utils.dump_json(bm25_rank_results, bm25_refine_output_path)
        except Exception as e:
            print(f'Failed to retrieve passages for {data}')
            print(f"Error: {e}")
    else: 
        logging.info(f"Loading local results for {data} from {bm25_refine_output_path}.")
        bm25_rank_results = utils.load_json(bm25_refine_output_path)
    return bm25_rank_results

def main(args):
    reranker = get_reranker(model_name = args.rank_model, mode = args.mode) 
    evaluation_results,evaluation_save_path = utils.evaluation_output_files(args) 
    
    for data in data_list:
        initial_retrieval_llm = 'vallina' if args.irmode == 'rerank' else args.llm 
        llm_rank_model_data = evaluation_results.get(initial_retrieval_llm, {}).get(args.rank_model)

        if llm_rank_model_data and data in llm_rank_model_data:
            logging.info(f"Skipping {data} since it's already evaluated in {'vallina' if args.rerank else args.llm} {args.rank_model}.")
            continue
        logging.info('#' * 20)
        logging.info(f'Evaluation on {data}')
        logging.info('#' * 20)
        # Retrieve or Loadding passages using pyserini BM25.
        
        if args.irmode == 'rerank':
            # rerank on valiina BM25 Top 100     
            searcher, topics, qrels = utils.get_data_pyserini(data,args.test)
            bm25_rank_results = run_retriever(topics, searcher, qrels, k=args.topk)
            rerank_result = reranker.rerank(bm25_rank_results,args.dense_topk)
            evaluation_results.setdefault('vallina', {}).setdefault(args.rank_model, {})
            evaluation_results['vallina'][args.rank_model][data] = utils.evalute_dict(rerank_result,benchmark.THE_TOPICS[data])
            utils.dump_json(evaluation_results, evaluation_save_path)
        else:
            #  Retrieve passages using BM25 with pseudo references
            bm25_rank_results = get_sparasemugi_result(data, args.test)

            if args.irmode == 'mugisparse':
                # Evaluate MuGI+BM25 
                if evaluation_results.get(f'{args.llm}', {}).get(data):
                    logging.info(f"Skipping {data} since it's already evaluated in bm25+mugi.")
                    continue
                bm25_rank_score = utils.evaluate_bm25(bm25_rank_results, benchmark.THE_TOPICS[data])
                logging.info(f'BM25 nDCG@10 on {data} is {bm25_rank_score}')
                evaluation_results.setdefault(f'{args.llm}', {})
                evaluation_results[f'{args.llm}'][data] = bm25_rank_score
                utils.dump_json(evaluation_results, evaluation_save_path)

            elif args.irmode in ['mugirerank', 'mugipipeline']:
                if evaluation_results.get(args.llm, {}).get(args.rank_model, {}).get(args.mode, {}).get(data):
                    logging.info(f"Skipping {data} since it's already evaluated in {args.llm} {args.rank_model} {args.mode}.")
                    continue
                use_enhanced_query = (args.irmode == 'mugipipeline')
                logging.info(f"Rerank top {args.dense_topk} documents on {data} using {args.rank_model} with{' enhanced query' if use_enhanced_query else ''}")
                gen_key = 'gen_cand_gpt35' if 'gpt' in args.llm else f'gen_cand_{args.llm}'
                rerank_result = reranker.rerank(bm25_rank_results, gen_key, args.dense_topk, use_enhanced_query=use_enhanced_query)

                # Ensure the nested dictionary structure exists
                evaluation_results.setdefault(args.llm, {}).setdefault(args.rank_model, {}).setdefault(args.mode, {})
                
                # Evaluate and update the evaluation results
                evaluation_results[args.llm][args.rank_model][args.mode][data] = utils.evalute_dict(rerank_result, benchmark.THE_TOPICS[data])
                
                # Save the evaluation results
                utils.dump_json(evaluation_results, evaluation_save_path)
            else:
                raise ValueError(f"Invalid mode: {args.ir}")



if __name__ == "__main__":
    args = parse_args()
    assert (args.repeat_times and not args.adaptive_times) or (not args.repeat_times and args.adaptive_times), "only assign times or adaptive_times"
    utils.display_args(args)
    data_list = ['dl20', 'dl19', 'covid', 'nfc' ,'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']
    main(args)

    