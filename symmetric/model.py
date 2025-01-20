import os
import torch
import torch.nn as nn
import numpy as np
import math
import random
import transformers
import pytorch_lightning as pl
import torch.distributed as dist
from transformers import AutoTokenizer, Adafactor, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import detect_anomaly

from data import ContrieverDataset
from src import roberta_contriever, contriever, utils, dist_utils

class FineTune_Contriever(pl.LightningModule):
    def __init__(self, hparams):
        super(FineTune_Contriever, self).__init__()

        self.save_hyperparameters(hparams)

        if "roberta" in hparams.model_name_or_path:
            self.model = roberta_contriever.Contriever.from_pretrained(hparams.model_name_or_path)
        else:
            assert "bert" in hparams.model_name_or_path or "contriever" in hparams.model_name_or_path
            self.model = contriever.Contriever.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token = "[CLS]" 
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = "[SEP]"

        self.model.config.pooling = "average" 
        self.run_stats = utils.WeightedAvgStats()

        self.validation_step_outputs = []
        # self.automatic_optimization = False
        # total_steps = ((len(self.train_dataloader())//self.hparams.n_gpu)+1)//self.hparams.gradient_accumulation_steps
        # self.optimizer, self.scheduler = utils.set_optim(self.hparams, self.model, total_steps) 

    def do_print(self, text):
        if torch.cuda.current_device() == 0:
            print(text)

    def train_dataloader(self):
        train_dataset = ContrieverDataset(
            datapaths=self.hparams.train_file,
            training=True,
            tokenizer=self.tokenizer,
            maxlength=self.hparams.max_length,
            negative_ctxs=self.hparams.negative_ctxs,
            negative_hard_ratio=self.hparams.negative_hard_ratio,
            negative_hard_min_idx=self.hparams.negative_hard_min_idx,
            normalize=self.hparams.eval_normalize_text,
        )
        self.do_print(f"# of training dataset: {len(train_dataset)}")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return train_dataloader

    def val_dataloader(self):
        eval_dataset = ContrieverDataset(
            datapaths=self.hparams.dev_file,
            training=False,
            tokenizer=self.tokenizer,
            maxlength=self.hparams.max_length,
            negative_ctxs=self.hparams.negative_ctxs,
            negative_hard_ratio=self.hparams.negative_hard_ratio,
            negative_hard_min_idx=self.hparams.negative_hard_min_idx,
            normalize=self.hparams.eval_normalize_text,
        )
        self.do_print(f"# of eval dataset: {len(eval_dataset)}")
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=self.hparams.train_batch_size, drop_last=False, num_workers=self.hparams.num_workers)
        return eval_dataloader

    def test_dataloader(self):
        eval_dataset = ContrieverDataset(
            datapaths=self.hparams.test_file,
            training=False,
            normalize=self.hparams.eval_normalize_text,
        )
        collator = Collator(self.tokenizer, maxlength=self.hparams.max_length)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.hparams.train_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collator,
        )
        return eval_dataloader

    def forward(self, q_tokens, q_mask, g_tokens, g_mask, n_tokens, n_mask, stats_prefix="", iter_stats={}, **kwargs):

        # print(f"q_tokens: {q_tokens.shape}")
        # print(f"q_mask: {q_mask.shape}")
        # print(f"g_tokens: {g_tokens.shape}")
        # print(f"g_mask: {g_mask.shape}")
        # print(f"n_tokens: {n_tokens.shape}")
        # print(f"n_mask: {n_mask.shape}")

        # with detect_anomaly():
        if self.hparams.negative_ctxs == 0:
            k_mask = g_mask
            k_tokens = g_tokens 
        else:
            k_mask = torch.cat([g_mask, n_mask], dim=0)
            k_tokens = torch.cat([g_tokens, n_tokens], dim=0) 
         
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.model(input_ids=q_tokens, attention_mask=q_mask, normalize=self.hparams.eval_normalize_text)
        kemb = self.model(input_ids=k_tokens, attention_mask=k_mask, normalize=self.hparams.eval_normalize_text)

        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)
        labels = labels + dist_utils.get_rank() * len(kemb)
        scores = torch.einsum("id, jd->ij", qemb / self.hparams.temperature, gather_kemb)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.hparams.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        self.log(f"{stats_prefix}accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stats_prefix}stdq", stdq, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stats_prefix}stdk", stdk, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # print(f"Loss!!! {loss}")

        return loss, iter_stats

    def training_step(self, batch, batch_idx):
        # dict_keys(['q_tokens', 'q_mask', 'k_tokens', 'k_mask', 'g_tokens', 'g_mask', 'n_tokens', 'n_mask'])
        # batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        
        self.model.train()
        #self.optimizer.zero_grad()

        if type(batch["n_tokens"]) != list and len(batch["n_tokens"].shape) == 3:
            dim = list(batch["n_tokens"].shape)[-1]
            batch["n_tokens"] = batch["n_tokens"].view(-1, dim)
            batch["n_mask"] = batch["n_mask"].view(-1, dim)
        
        train_loss, iter_stats = self(**batch, stats_prefix="train")
        self.run_stats.update(iter_stats)

        #self.manual_backward(train_loss)
        #self.optimizer.step()
        #self.scheduler.step()

        return train_loss

    def validation_step(self, batch, batch_idx):
        
        self.model.eval()
        # batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
       
        if self.hparams.dev_negative_ctxs == 0:
            all_tokens = batch["g_tokens"]
            all_mask = batch["g_mask"]
        else:
            all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
            all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)

        q_emb = self.model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=self.hparams.eval_normalize_text)
        all_emb = self.model(input_ids=all_tokens, attention_mask=all_mask, normalize=self.hparams.eval_normalize_text)

        g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

        ret = {"q_emb": q_emb, "g_emb": g_emb, "n_emb": n_emb}
        self.validation_step_outputs.append(ret)
        return

    def on_validation_epoch_end(self):

        all_q = torch.cat([x["q_emb"] for x in self.validation_step_outputs], dim=0)
        all_g = torch.cat([x["g_emb"] for x in self.validation_step_outputs], dim=0)
        all_n = torch.cat([x["n_emb"] for x in self.validation_step_outputs], dim=0)

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
        scores = torch.cat([scores_pos, scores_neg], dim=-1)

        argmax_idx = torch.argmax(scores, dim=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None]
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0)
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        self.log('val_eval_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log('val_eval_mrr', mrr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            print(" | ".join(message))
            # if tb_logger is not None:
            #     tb_logger.add_scalar(f"eval_acc", acc, step)
            #     tb_logger.add_scalar(f"mrr", mrr, step)
        self.validation_step_outputs = []
        save_path = os.path.join(self.hparams.output_dir, f"best_tfmr_{self.current_epoch}")
        self._save_checkpoint()

    def test_step(self, batch, batch_idx):
        batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
        all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)

        q_emb = self.model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=self.hparams.eval_normalize_text)
        all_emb = self.model(input_ids=all_tokens, attention_mask=all_mask, normalize=self.hparams.eval_normalize_text)

        g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

        ret = {"q_emb": q_emb, "g_emb": g_emb, "n_emb": n_emb}
        self.validation_step_outputs.append(ret)
        return

    def on_test_epoch_end(self):

        all_q = torch.cat([x["q_emb"] for x in self.validation_step_outputs], dim=0)
        all_g = torch.cat([x["g_emb"] for x in self.validation_step_outputs], dim=0)
        all_n = torch.cat([x["n_emb"] for x in self.validation_step_outputs], dim=0)

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
        scores = torch.cat([scores_pos, scores_neg], dim=-1)

        argmax_idx = torch.argmax(scores, dim=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None]
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0)
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        self.log('val_eval_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_eval_mrr', mrr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            print(" | ".join(message))
            # if tb_logger is not None:
            #     tb_logger.add_scalar(f"eval_acc", acc, step)
            #     tb_logger.add_scalar(f"mrr", mrr, step)
        self.validation_step_outputs.clear()

    def _set_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, warmup_init=False, scale_parameter=False, relative_step=False,)
        return optimizer

    def configure_optimizers(self):
        model = self.model
        
        # for name, para in model.named_parameters():
        #     print(f"[{name}] -> {para.requires_grad}")
        # import sys; sys.exit()


        # total_steps = ((len(self.train_dataloader())//self.hparams.n_gpu)+1)//self.hparams.gradient_accumulation_steps
        # self.optimizer, self.scheduler = utils.set_optim(self.hparams, self.model, total_steps) 
        # return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step', 'name': 'learning_rate'}] 

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, warmup_init=False, scale_parameter=False, relative_step=False,)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999) #, eps=opt.eps, weight_decay=opt.weight_decay
        )
        self.opt = optimizer 
       
        if self.hparams.lr_scheduler == 'linear':
            self.do_print("*** learning rate scheduler is CONSTANT")
            return [optimizer]
            # return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'learning_rate'}] 
        elif self.hparams.lr_scheduler == "exponential":
            self.do_print("*** learning rate scheduler is EXPONENTIAL")
            len_data = len(self.train_dataloader())
            denominator=self.hparams.n_gpu
            steps_per_epoch=((len_data//denominator)+1)//self.hparams.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy= 'linear', cycle_momentum=False)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'learning_rate'}] 
        else:
            assert False

    def _save_checkpoint(self):
        save_path = os.path.join(self.hparams.output_dir, f"best_tfmr_{self.current_epoch}")
        self.do_print(f"Save.. {self.current_epoch} in {save_path}")
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
