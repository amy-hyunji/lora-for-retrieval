# late-interaction
Modified code from [official Colbert repository](https://github.com/stanford-futuredata/ColBERT). We experiment with Colbert v1 (`colbertv1` branch) as the training code is released. 

### 1. Create conda environment and Install requirements
```
conda env create -f conda_env.yml
conda activate colbert-v0.2
```

### 2. Train
- full tuning (FT)
```
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch --nproc_per_node=4 -m \
colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples /path/to/MSMARCO/triples.train.small.tsv \
--root /root/to/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
```
- Lora
```
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch --nproc_per_node=4 -m \
colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --lora \
--triples /path/to/MSMARCO/triples.train.small.tsv \
--root /root/to/experiments/ --experiment MSMARCO-psg-lora --similarity l2 --run msmarco.psg.l2
```

### 3. Inference
We utilize code from [beir-ColBERT](https://github.com/thakur-nandan/beir-ColBERT) for evaluation over beir benchmark.
```
git clone https://github.com/thakur-nandan/beir-ColBERT.git
bash evaluate_beir.sh
```
