# MuGI: Multi-Text Generation Intergration for IR

Code for paper [Exploring the Best Practices of Query Expansion with Large Language Models]([https://arxiv.org/abs/2401.06311](https://aclanthology.org/2024.findings-emnlp.103/))

This project aims to explore generated documents for enhanced IR with LLMs. We enhance BM25 to surpass strong dense retriever on many datasets.

***MuGI surpasses all current query-rewriting methods, enhancing BM25 and outperforming E5-Mistral-7b-instruct in out-of-distribution (OOD) evaluations. Furthermore, it boosts a 110M model to achieve state-of-the-art (SOTA) results in in-distribution (IID) assessments**.*

![Screenshot 2024-01-22 at 12.56.51 PM](https://github.com/lezhang7/Retrieval_MuGI/blob/main/asset/performance.png)


## News

- **[2024.01.12]** Our paper is now available at https://arxiv.org/abs/2401.06311

## Quick example

Install requirements

```python
pip install -r requirements.txt
```

Below defines a query:

```python
topics={ 264014: {'title': "how long is life cycle of flea"}}
```

We can generate multiple passages using ChatGPT with instructional permutation generation:

```python
from mugi import generate_pseudo_references
from openai import OpenAI
import os
openai_key =  os.environ.get('OPENAI_KEY')
client = OpenAI(api_key=openai_key)
new_topics = generate_pseudo_references(topics, generated_document_num=3)
```

We get the following result:

```python
{ 264014: {'title': "how long is life cycle of flea",
    'gen_cand_gpt4': [
            "The life cycle of a flea typically spans 3 to 4 weeks, but it can extend to several months under less than ideal conditions. The cycle is composed of four stages: egg, larva, pupa, and adult. The initial stage begins when an adult female lays eggs after feeding on a host, often a mammal or bird. These eggs hatch into larvae within a few days, which then feed on organic debris found in their environment. After several molts, larvae develop into pupae, encapsulating themselves in cocoons for approximately one to two weeks. Finally, adult fleas emerge ready to find a host, feed, and begin the reproductive cycle anew. Environment and climate can influence the duration of this cycle, with warmth and humidity generally accelerating the process.",
            "The life cycle of a flea comprises four stages: egg, larva, pupa, and adult. This process can take anywhere from two weeks to several months, depending on environmental conditions. The cycle begins when an adult flea lays eggs on a host. These eggs can hatch into larvae within a couple of days under optimal conditions, which include warmth and high humidity. Larvae feed on organic debris and adult flea feces for several days before spinning cocoons to enter the pupal stage. Pupation may last from a few days to several weeks or months; during this time, the pupa awaits environmental cues signaling the presence of a potential host. Finally, adult fleas emerge from the cocoon, ready to feed and restart the cycle. While the ideal temperature for the flea's life cycle is between 70-85\u00b0F (21-29\u00b0C) with high humidity, cooler or drier conditions can extend the cycle significantly, with pupae lying dormant for up to a year in unfavorable conditions.",
            "Title: The Flea Life Cycle\n\nThe life cycle of a flea encompasses four stages: egg, larva, pupa, and adult. The duration of a flea's life cycle can vary from a couple weeks to several months, depending on environmental conditions. \n\n1. Egg: The cycle begins when an adult female flea lays eggs after feeding on a host. These eggs are commonly dispersed in the environment where the host lives and sleeps.\n   \n2. Larva: In 2-10 days, eggs hatch into larvae which avoid light and feed on organic debris found in their surroundings, including feces of adult fleas which contain undigested blood.\n\n3. Pupa: After several weeks, the larvae spin silk-like cocoons, entering the pupal stage. Pupae can remain dormant for months, waiting for a signal (e.g., warmth, CO2, or vibrations) to emerge as adults.\n\n4. Adult: Triggered by the right conditions, adults emerge from the pupae. Within hours, they will seek a host to feed on blood, mate, and begin laying eggs, starting the cycle over again.\n\nIn optimal conditions, the entire flea life cycle can be as short as 3 weeks, but if conditions are not conducive, it can extend to a year or more. Control measures must target all stages of the life cycle to effectively eliminate fleas."
        ],
        'gen_cand_gpt35': [
            "The life cycle of a flea typically lasts about 30 to 75 days, although it can vary depending on environmental conditions such as temperature and humidity. Fleas go through four main stages in their life cycle: egg, larva, pupa, and adult. The entire life cycle can be as short as two weeks or as long as eight months under favorable conditions.",
            "The life cycle of a flea typically lasts about 30-75 days, but can vary depending on environmental conditions such as temperature and humidity. Fleas go through four stages in their life cycle: egg, larva, pupa, and adult. From egg to adult, the entire life cycle can take as little as two weeks or as long as several months.",
            "The life cycle of a flea typically consists of four stages: egg, larva, pupa, and adult. The entire life cycle can range from as little as a few weeks to as long as several months, depending on environmental conditions such as temperature and humidity. Under ideal conditions, the life cycle can be completed in as little as 14 days, but it can be extended significantly in less favorable environments."
        ]
      }
}
```

we can perform **information retrieval with BM25** with:

```python
from mugi import run_retriever_mugi
import utils
from pyserini.search import LuceneSearcher, get_topics, get_qrels
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
qrels = get_qrels('dl19-passage')
repetition_ratio=5
articles_num=3
for key in topics:
      query = topics[key]['title']
      gen_ref = ' '.join(topics[key]['gen_cand_gpt4'][:articles_num])
      repetition_times = (len(gen_ref)//len(query))//repetition_ratio
      topics[key]['enhanced_query'] = (query + ' ')*repetition_times + gen_ref
bm25_rank_results = run_retriever_mugi(topics, searcher, qrels, k=100)
# eval nDCG@10
bm25_rank_score=utils.evaluate_bm25(bm25_rank_results,'dl19-passage')
print(bm25_rank_score)
```

or **re-rank using dense retrieval models**:

```python
from model import get_language_model, get_reranker
reranker_model = get_reranker(model_name = 'all-mpnet-base-v2', mode = 'concat')
rerank_result = reranker_model.rerank(bm25_rank_results,'gen_cand_gpt35',100,use_enhanced_query=True)
rerank_score = utils.evalute_dict(rerank_result,'dl19-passage')
print(rerank_score)
```

## Evaluation on Benchmarks
Data released at: [🤗 Hub](https://huggingface.co/datasets/le723z/mugi/tree/main)

```python
pip install --upgrade --no-cache-dir gdown # must update gdown to avoid bugs, thanks to https://github.com/wkentaro/gdown/issues/146
bash download.sh   # Download GPT generated documents  
```

We have 4 `irmode` of applying MuGI including `['mugisparse','rerank','mugirerank','mugipipeline']`, to run MuGI:

```python
python mugi.py --llm gpt --irmode $irmode 
```

To **generated refences with open-source LLMs**, selectiong  `llm` in `[01-ai/Yi-34B-Chat-4bits, 01-ai/Yi-6B-Chat-4bits, Qwen/Qwen1.5-7B-Chat-AWQ, Qwen/Qwen1.5-14B-Chat-AWQ, Qwen/Qwen1.5-72B-Chat-AWQ]*`,and run

```
python mugi.py --llm $llm --irmode $irmode 
```



## Cite

```latex
@article{zhang2024exploring,
  title={Exploring the Best Practices of Query Expansion with Large Language Models},
  author={Zhang, Le and Wu, Yihong and Yang, Qian and Nie, Jian-Yun},
  journal={arXiv preprint arXiv:2401.06311},
  year={2024}
}
```
