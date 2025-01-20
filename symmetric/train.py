import json
import random
import torch
import argparse
import textwrap

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def do_print(text):
    if torch.cuda.current_device() == 0:
        print(text)

def main(args, train_params):
    set_seed(args.seed)
    print(f"peft setting ... [{args.peft}]")
    if args.peft == "lora":
        from peft_model import FineTune_Contriever 
        model = FineTune_Contriever(args)
    elif args.peft == "prompt_tuning":
        from pt_model import FineTune_Contriever
        model = FineTune_Contriever(args)
    else:
        from model import FineTune_Contriever 
        model = FineTune_Contriever(args)
    trainer = pl.Trainer(**train_params)
    if args.do_train:
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
        print("####### Done training")
    if args.do_test:
        trainer.test(model)
        print("###### Done testing")
    return args.output_dir

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NotImplementedError("Input Config File is Needed!")

    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    if hparam.do_train:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name)
    else:
        wandb_logger = None

    # Set Configurations
    args_dict = dict(
        output_dir=hparam.output_dir,
        model_name_or_path=hparam.model_name_or_path,
        tokenizer_name_or_path=hparam.tokenizer_name_or_path,
        max_length=hparam.max_length,
        learning_rate=hparam.learning_rate, # 0.001 for finetune
        lr_scheduler=hparam.lr_scheduler,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        num_train_epochs=hparam.num_train_epochs, # 20,000
        train_batch_size=hparam.train_batch_size, # 8
        eval_batch_size=hparam.eval_batch_size,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.n_gpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint,
        lora_ckpt=hparam.lora_ckpt if "lora_ckpt" in hparam else None,
        val_check_interval=0.05,
        early_stop_callback=False,
        fp_16=False,
        seed=hparam.seed,
        check_val_every_n_epoch=hparam.check_val_every_n_epoch,
        train_file=hparam.train_file,
        dev_file=hparam.dev_file,
        test_file=hparam.test_file,
        corpus_file=hparam.corpus_file,
        do_train=hparam.do_train,
        do_test=hparam.do_test,
        negative_ctxs=hparam.negative_ctxs,
        dev_negative_ctxs=0,
        negative_hard_ratio=hparam.negative_hard_ratio,
        negative_hard_min_idx=hparam.negative_hard_min_idx,
        eval_normalize_text=hparam.eval_normalize_text,
        label_smoothing=hparam.label_smoothing,
        temperature=hparam.temperature,
        log_freq=hparam.log_freq,
        peft=hparam.peft,
        debug=hparam.debug if "debug" in hparam else False
        # lora=hparam.lora if "lora" in hparam else False
    )
    args = argparse.Namespace(**args_dict)

    do_print(args)
    do_print(f"** Output Dir {args.output_dir}")

    # Define checkpoint function
    #checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, dirpath=args.output_dir, filename='{epoch:02d}-{val_loss:.2f}')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                monitor="val_eval_acc", 
                                mode="max", 
                                dirpath=args.output_dir, 
                                save_top_k=5, 
                                filename='{epoch:02d}-{val_eval_acc:.2f}'
                                )

    # Logging Learning Rate Scheduling
    if args.learning_rate == "linear":
        do_print("@@@ Not using learning rate scheduler")
        lr_monitor = []
    else:
        lr_monitor = [pl.callbacks.LearningRateMonitor()]

    if args.debug:
        print(f"## is DEBUG state!")
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            devices=args.n_gpu,
            max_epochs=args.num_train_epochs,
            precision="bf16" if args.fp_16 else 32,
            callbacks=lr_monitor+[checkpoint_callback],
            val_check_interval=0.05,
            logger=wandb_logger,
            accelerator="gpu",
            strategy="ddp",
            #check_val_every_n_epoch=args.check_val_every_n_epoch,
            log_every_n_steps=800,
        )
    else:
        print(f"## not DEBUG state!")
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            devices=args.n_gpu,
            max_epochs=args.num_train_epochs,
            precision="bf16" if args.fp_16 else 32,
            callbacks=lr_monitor+[checkpoint_callback],
            val_check_interval=1.0,
            logger=wandb_logger,
            accelerator="gpu",
            strategy="ddp",
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            #log_every_n_steps=200,
        )
    main(args, train_params)
