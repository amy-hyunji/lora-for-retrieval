# simple-recipe-to-improve-OOD

This repository contains the code for ```[A Simple Recipe for Improving Out-of-Domain Retrieval in Dense Encoders](https://arxiv.org/abs/2311.09765)```

![alt text](fig/fig1.png "Main Figure")

We recommend a simple recipe for training dense encoders to improve out-of-domain performance:
1. Train with parameter-efficient methods such as LoRA rather than full-finetuning
2. opt for using in-batch negatives unless given well-constructed hard negatives.

We conduct a series of carefully designed experiments over the recipes and could see that our findings hold for various cases such as adopting larger base models, different retriever architectures (e.g., late interaction models), and additional contrastive pretraining of the base model (e.g., pre-trained Contriever)

## Requirements
```
pip install -r requirements.txt
```

## Dataset
### Train Dataset
* [MSMARCO](https://microsoft.github.io/msmarco/)
* [NQ](https://github.com/facebookresearch/DPR)
### Test Dataset
* [Beir](https://github.com/beir-cellar/beir)

### Negative Sampling
* [MSMARCO](https://microsoft.github.io/msmarco/)
* Random
* BM25
* self-distillation
* self-distillation with denoising step
* [RocketQA](https://github.com/PaddlePaddle/RocketQA)

## Train / Inference

We divide the files by the dense retriever architectures we experiment over: asymmetric, symmetric, and late interaction.
For each, the code base is from [DPR](https://github.com/facebookresearch/DPR), [contriever](https://github.com/facebookresearch/contriever), and [Colbert](https://github.com/stanford-futuredata/ColBERT), as we follow the architectural design from each model. 

Details of how to run the models are under each folder:
* [asymmetric](https://github.com/amy-hyunji/simple-recipe-to-improve-OOD/blob/main/asymmetric/README.md)
* [symmetric](https://github.com/amy-hyunji/simple-recipe-to-improve-OOD/blob/main/symmetric/README.md)
* [late-interaction](https://github.com/amy-hyunji/simple-recipe-to-improve-OOD/blob/main/late-interaction/README.md)
