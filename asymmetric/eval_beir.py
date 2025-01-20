from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "climate-fever"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

print(dataset)
multiset = False 
print(f"is multiset: {multiset}")
#### Load the SBERT model and retrieve using cosine-similarity
if multiset:
    model = DRES(models.DPR(["facebook/dpr-question_encoder-multiset-base", "facebook/dpr-ctx_encoder-multiset-base"]), batch_size=8)
else:
    model = DRES(models.DPR(["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-ctx_encoder-single-nq-base"]), batch_size=8)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
# ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
print(retriever.evaluate(qrels, results, retriever.k_values))