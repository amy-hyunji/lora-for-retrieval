# Symmetric
Modified code from [official Contriever repository](https://github.com/facebookresearch/contriever)

### 1. Create conda environment and Install requirements
```
conda create -n sy python=3.7 && conda activate sy
pip install -r requirements.txt
```

### 2. Set config
details of configuration in `config/`

### 3. Train
- full tuning (FT)
```
python train.py --config config/bert_base.full.wo_neg.json
```
- Lora
```
python train.py --config config/bert_base.lora.wo_neg.json
```

### 4. Inference
```
python ./eval_beir.py --model_name_or_path $MODELPATH --dataset $DATAPATH --base_model bert-base --per_gpu_batch_size 1024
```
