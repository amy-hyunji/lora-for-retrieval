# LoRA for Retrieval

This repository contains the code for our work on LoRA for retrieval. This work was conducted while Hyunji Lee was an intern at the Allen Institute for AI.

### Setup

To create the environment, run:

```
conda create -n 'lora-for-retrieval' python=3.8
conda activate lora-for-retrieval
```

To install dependencies, run:
```
pip install -e .'[dev]'
```

To run tests, run:
```
pytest tests
```


### Usage

TBD code coming soon!

### Learn more about our work

Check out <a href="https://github.com/amy-hyunji/lora-for-retrieval/tree/main/paper/hyunji-lee-back-to-basics-simple-recipes-for-ood-retrieval.pdf">our paper</a>. If you find our work useful, please cite us:

```
@article{hyunji-lee-lora-retrieval-2023,
  title={Back to Basics: A Simple Recipe for Improving Out-of-Domain Retrieval in Dense Encoders},
  author={Lee, Hyunji and Soldaini, Luca Soldaini and Cohan, Arman and Seo, Minjoon and Lo, Kyle},
  journal={ArXiv},
  year={2023}
}
```