import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
# from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]
        item_ids_raw = [d["item_ids"] for d in batch]
        cf_input_texts = ["" for d in batch]
        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100


        inputs_cf = self.tokenizer(cf_input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs["input_ids_cf"] = inputs_cf["input_ids"]
        inputs["attention_mask_cf"] = inputs_cf["attention_mask"]

        # --- item_ids for item frequency weighting ---
        # === build item_ids tensor, -1 denotes padding(neglect) ===
        B, L = inputs["labels"].shape
        item_ids = torch.full((B, L), -1, dtype=torch.long)         # (B,L)
        for i, iid in enumerate(item_ids_raw):
            seq_len = labels["attention_mask"][i].sum()             
            item_ids[i, :seq_len] = iid
        inputs["item_ids"] = item_ids   

        return inputs



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        other_targets = [d["other_labels"] for d in batch]
        target_ids = [d["item_ids"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets, target_ids, other_targets)

