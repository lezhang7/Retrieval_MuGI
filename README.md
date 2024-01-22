# MuGI: Multi-Text Generation Intergration for IR

Code for paper "[[MuGI: Enhancing Information Retrieval through Multi-Text Generation Intergration with Large Language Models](https://arxiv.org/abs/2401.06311)](https://arxiv.org/abs/2401.06311)"

This project aims to explore generated documents for enhanced IR with LLMs.


## News

- **[2024.01.12]** Our paper is now available at https://arxiv.org/abs/2401.06311

## Quick example

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
for key in topics:
      query = topics[key]['title']
      gen_ref = ' '.join(topics[key]['gen_cand_gpt4'][:3])
      repetition_times = (len(gen_ref)//len(query))//5
      topics[key]['enhanced_query'] = (query + ' ')*repetition_times + gen_ref
bm25_rank_results = run_retriever_mugi(topics, searcher, qrels, k=100)
# eval nDCG@10
rank_score=utils.evaluate_bm25(bm25_rank_results)
print(rank_score)
```

or **re-rank using dense retrieval models**:

```
from model import get_language_model, get_reranker
reranker_model = get_reranker(model_name = 'all-mpnet-base-v2', mode = 'concat')
rerank_result = reranker_model.rerank(bm25_rank_results,100)
rerank_score = utils.evalute_dict(rerank_result,'dl19-passage')
print(rerank_score)
```



## Evaluation on Benchmarks

1. **Download generated documents by running**

   `bash download.sh`

2. Run evaluation on all benchmarks (modify `data_list` in mugi.py )

​	`		python mugi.py --mode concat `

Below are the results (average nDCG@10) of our preliminary experiments 

### ![Screenshot 2024-01-22 at 12.56.51 PM](https://p.ipic.vip/4fkjyz.png)

## Cite

```latex
@misc{zhang2024mugi,
      title={MuGI: Enhancing Information Retrieval through Multi-Text Generation Intergration with Large Language Models}, 
      author={Le Zhang and Yihong Wu},
      year={2024},
      eprint={2401.06311},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```