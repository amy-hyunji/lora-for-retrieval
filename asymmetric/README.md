# Asymmetric (`asymmetric/`)
Modified code from [official DPR repository](https://github.com/facebookresearch/DPR)

### 1. Create conda environment and Install requirements
```
conda create -n asy python=3.7 && conda activate asy
pip install -r requirements.txt
```

### 2. Set config
details of configuration in `conf/`

### 3. Train
- full tuning (FT)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 1020 train_dense_encoder.py train=biencoder_nq encoder=hf_bert train_datasets=[msmarco_train] dev_datasets=[msmarco_dev] output_dir="./dpr/msmarco_randomneg_full"
```
- Lora
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 1020 train_dense_encoder.py train=biencoder_nq encoder=hf_lora_bert train_datasets=[msmarco_train_randomneg] dev_datasets=[msmarco_dev] output_dir="./dpr/msmarco_randomneg_full"
```

### 4. Inference
- generate dense embeddings
```
python generate_dense_embeddings.py model_file=./dpr/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp shard_id=0 num_shards=1 out_file=./outputs/dpr_single/dpr_single_msmarco ctx_src=beir_msmarco encoder=hf_bert batch_size=2048
```
- do retrieval. scores in `./outputs/result.json`
```
python dense_retriever.py model_file=./dpr/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp qa_dataset=beir_msmarco_test ctx_datatsets=beir_msmarco	encoded_ctx_files="./outputs/dpr_single/dpr_single_msmarco_0" out_file=./outputs/result.json n_docs=100;
```
