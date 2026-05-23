import copy
import torch


class Collator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.need_cf = "cft" in str(getattr(args, "method", "")).lower()
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        # Causal-LM training targets should be built from the full prompt+response text,
        # then mask out the prompt part to mimic seq2seq-style target-only supervision.
        inputs = self.tokenizer(
            text=full_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        prompt_inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="max_length",
            max_length=inputs["input_ids"].shape[1],
            truncation=True,
            return_attention_mask=True,
        )

        labels = copy.deepcopy(inputs["input_ids"])
        # Keep true EOS trainable even when pad_token_id == eos_token_id (common in LLaMA).
        labels[inputs["attention_mask"] == 0] = -100
        if self.only_train_response:
            prompt_mask = prompt_inputs["attention_mask"].bool()
            labels[prompt_mask] = -100
        inputs["labels"] = labels

        if self.need_cf:
            cf_input_texts = [d.get("cf_input_ids", "") for d in batch]
            cf_full_texts = [d.get("cf_labels", d["labels"]) + self.tokenizer.eos_token for d in batch]

            cf_inputs = self.tokenizer(
                text=cf_full_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=inputs["input_ids"].shape[1],
                truncation=True,
                return_attention_mask=True,
            )
            cf_prompt_inputs = self.tokenizer(
                text=cf_input_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=inputs["input_ids"].shape[1],
                truncation=True,
                return_attention_mask=True,
            )

            if self.only_train_response:
                # Keep masked labels for optional debugging/inspection.
                cf_labels = copy.deepcopy(cf_inputs["input_ids"])
                cf_labels[cf_inputs["attention_mask"] == 0] = -100
                cf_labels[cf_prompt_inputs["attention_mask"].bool()] = -100
                inputs["labels_cf"] = cf_labels

            inputs["input_ids_cf"] = cf_inputs["input_ids"]
            inputs["attention_mask_cf"] = cf_inputs["attention_mask"]

        item_ids_raw = [d.get("item_ids", -1) for d in batch]
        bsz, seq_len = inputs["labels"].shape
        item_ids = torch.full((bsz, seq_len), -1, dtype=torch.long)
        for i, iid in enumerate(item_ids_raw):
            if iid is None:
                continue
            valid_pos = (inputs["labels"][i] != -100).nonzero(as_tuple=False).view(-1)
            if valid_pos.numel() == 0:
                continue
            item_ids[i, valid_pos] = int(iid)
        inputs["item_ids"] = item_ids

        return inputs


class TestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # Decoder-only generation should use left padding.
        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        other_targets = [d.get("other_labels", None) for d in batch]
        target_ids = [d.get("item_ids", None) for d in batch]
        users = [d.get("user", None) for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets, target_ids, other_targets, users)
