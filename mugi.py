
from tqdm import tqdm
from pyserini.search import LuceneSearcher, get_topics, get_qrels
import os
import json
import utils
from typing import List, Dict, Tuple, Tuple, Optional
import argparse
import logging
from model import get_language_model, get_reranker
import benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

# set up chatgpt api
def run_gpt(message,model_name='gpt-3.5-turbo'):
    response = client.chat.completions.create(model=model_name,messages=message)
    return response.choices[0].message.content

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rerank', action='store_true', help='whether to rerank on original bm25')
    # documents generation setting
    parser.add_argument('--gen_model', type=str, default='openai', help='pseudo reference generation model')
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
    parser.add_argument('--mode', type=str, choices=['query', 'alternate', 'concat','qg'],required=True, help='whether to use generated reference')
    
    


    args = parser.parse_args()
    return args

def generate_pseudo_references(topics:Dict[str,Dict[str,str]], generated_document_num:int):
    """
    Generate pseudo references for the given data in the form of a list of item pairs. Save to output_path.
    Example:
        topics={query_id:{"title":query_text}}
    Output: 
        topics={query_id:{"title":query_text,"gen_cand_gpt4":[<gen_doc1>,<gen_doc2>,...],"gen_cand_gpt35":[<gen_doc1>,<gen_doc2>,...]}}
    """
    for key in tqdm(topics):
        query=topics[key]['title']
        message=utils.PromptTemplate.get_prefix_prompt_pseudo_doc_zs(query)
        for _ in range(generated_document_num):
            output=run_gpt(message,model_name='gpt-4-1106-preview').strip()
            if 'gen_cand_gpt4' in topics[key]:
                topics[key]['gen_cand_gpt4'].append(output)
            else:
                topics[key]['gen_cand_gpt4']=[output]
        for _ in range(min(3,generated_document_num)):
            output=run_gpt(message,model_name='gpt-3.5-turbo-1106').strip()
            if 'gen_cand_gpt35' in topics[key]:
                topics[key]['gen_cand_gpt35'].append(output)
            else:
                topics[key]['gen_cand_gpt35']=[output]
    return topics

def bm25_with_psuedo_ref(args, topics, searcher, qrels, data):
    """
    Perform bm25 with generated pseudo references by gpt.
    Concate the query and pseudo reference in the form of <q>*t+<ref>
    """
    # 1. Generate pseudo references 
    logging.info(f'Generating pseudo references for {data}')
    topics = generate_pseudo_references(topics, args.doc_gen)

    # 2. Run bm25 with pseudo references
    for key in topics:
        query = topics[key]['title']
        gen_ref = ' '.join(topics[key]['gen_cand_gpt4'][:args.article_num])
        if args.repeat_times:
            times = args.repeat_times
        elif args.adaptive_times:
            times = (len(gen_ref)//len(query))//args.adaptive_times
        topics[key]['enhanced_query'] = (query + ' ')*times + gen_ref
    rank_results = run_retriever_mugi(topics, searcher, qrels, k=args.topk)

    return rank_results

def run_retriever_mugi(topics, searcher, qrels=None, k=100, qid=None):
    """
    Run retriever on a list of topics.  If qrels is provided, only run on topics that are in qrels.
    Used as BM25 baseline.
    """
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['enhanced_query']
            ranks.append({'query': topics[qid]['title'], 'hits': [], 'gen_cand_gpt4': topics[qid]['gen_cand_gpt4'],'gen_cand_gpt35': topics[qid]['gen_cand_gpt35']})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks
def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    """
    Run retriever on a list of topics.  If qrels is provided, only run on topics that are in qrels.

    Used as BM25 baseline.
    """
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def main(args):
    data_list = ['dl20', 'dl19', 'covid', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']
    reranker_model = get_reranker(model_name = args.rank_model, mode = 'query' if args.rerank else args.mode) 
    evaluation_save_path = os.path.join('results',f"evaluation_results_{'rerank' if args.rerank else args.mode}.json")
    if os.path.exists(evaluation_save_path):
        logging.info(f"Loading evaluation results from {evaluation_save_path}")
        evaluation_results = utils.load_json(evaluation_save_path)
        if args.rank_model not in evaluation_results:
            evaluation_results[args.rank_model] = {}
    else:
        evaluation_results = {args.rank_model:{}}
    
    for data in data_list:
        if data in evaluation_results[args.rank_model]:
            logging.info(f"Skipping {data} since it's already evaluated.")
            continue
        logging.info('#' * 20)
        logging.info(f'Evaluation on {data}')
        logging.info('#' * 20)
        # Retrieve or Loadding passages using pyserini BM25.
        
        if args.rerank:
            # rerank on valiina BM25 Top 100     
            searcher = LuceneSearcher.from_prebuilt_index(benchmark.THE_INDEX[data])
            topics = get_topics(benchmark.THE_TOPICS[data] if data != 'dl20' else 'dl20')
            qrels = get_qrels(benchmark.THE_TOPICS[data])
            topics = {k: v for k, v in topics.items() if k in qrels}
            rank_results = run_retriever(topics, searcher, qrels, k=args.topk)
            rerank_result = reranker_model.rerank(rank_results,args.dense_topk)
            evaluation_results[args.rank_model][data] = utils.evalute_dict(rerank_result,benchmark.THE_TOPICS[data])
            logging.info(f"Saving evaluation results to {evaluation_save_path}")
            utils.dump_json(evaluation_results, evaluation_save_path)
        else:
            # rerank on MuGI+BM25 Top 100  
            bm25_refine_output_path = os.path.join(args.output_path, data+'_bm25_refine.json')
            if not os.path.exists(bm25_refine_output_path):
                logging.info(f"No local results found for {data}, generating psuedo references and retrieve passages using pyserini BM25, saving to {bm25_refine_output_path}.")
                try:
                    searcher = LuceneSearcher.from_prebuilt_index(benchmark.THE_INDEX[data])
                    topics = get_topics(benchmark.THE_TOPICS[data] if data != 'dl20' else 'dl20')
                    qrels = get_qrels(benchmark.THE_TOPICS[data])
                    topics = {k: v for k, v in topics.items() if k in qrels} #only run on topics that are in qrels
                    if args.test:
                        topics = {key:topics[key] for key in list(topics)[:2]}
                    rank_results = bm25_with_psuedo_ref(args, topics, searcher, qrels, data)
                    utils.dump_json(rank_results, bm25_refine_output_path)
                    
                except Exception as e:
                    print(f'Failed to retrieve passages for {data}')
                    print(f"Error: {e}")
                    continue
            else: 
                logging.info(f"Loading local results for {data} from {bm25_refine_output_path}.")
                rank_results = utils.load_json(bm25_refine_output_path)


            bm25_rank_score = utils.evaluate_bm25(rank_results, benchmark.THE_TOPICS[data])
            logging.info(f'BM25 nDCG@10 on {data} is {bm25_rank_score}')

            # Dense retrieval 
            logging.info(f'Rerank top {args.dense_topk} documents on {data} using {args.rank_model}')
            rerank_result = reranker_model.rerank(rank_results,args.dense_topk)
            # Evaluate nDCG@10
            evaluation_results[args.rank_model][data] = utils.evalute_dict(rerank_result,benchmark.THE_TOPICS[data])
            # Rename the output file to a better name
            utils.dump_json(evaluation_results, evaluation_save_path)



if __name__ == "__main__":
    args = parse_args()
    assert (args.repeat_times and not args.adaptive_times) or (not args.repeat_times and args.adaptive_times), "only assign times or adaptive_times"
    utils.display_args(args)

    if 'openai' in args.gen_model:
        from openai import OpenAI
        import os
        openai_key =  os.environ.get('OPENAI_KEY')
        client = OpenAI(api_key=openai_key)
    main(args)

    