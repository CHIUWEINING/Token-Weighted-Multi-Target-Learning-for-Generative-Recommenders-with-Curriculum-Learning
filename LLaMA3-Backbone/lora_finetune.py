import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import transformers
from typing import List
from transformers import TrainerCallback, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# import wandb
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from modeling_letter import LETTER
# from fastchat.train.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, LlamaConfig
from safetensors.torch import load_file as safetensors_load_file

from utils import *
from collator import Collator
from igd_precompute import IGComputer

LOG_VARS_FILE = "log_vars.pt"


def _unwrap_trainable_model(model):
    """Unwrap DDP/compile wrappers and PEFT wrapper to reach the LETTER module."""
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "_orig_mod"):
        base = base._orig_mod
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        return base.base_model.model
    return base


def _get_log_vars_param(model):
    base = _unwrap_trainable_model(model)
    return getattr(base, "log_vars", None)


def _ensure_log_vars_trainable(model):
    log_vars = _get_log_vars_param(model)
    if log_vars is None:
        return
    log_vars.requires_grad = True


def _sanitize_log_vars(model, clamp_abs=20.0):
    """
    Keep log_vars finite and within a conservative bound.
    This is a no-op in normal training and only activates on abnormal values.
    """
    log_vars = _get_log_vars_param(model)
    if log_vars is None:
        return False

    with torch.no_grad():
        has_non_finite = not torch.isfinite(log_vars).all()
        out_of_range = bool((log_vars > clamp_abs).any() or (log_vars < -clamp_abs).any())
        if not (has_non_finite or out_of_range):
            return False

        log_vars.nan_to_num_(nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
        log_vars.clamp_(min=-clamp_abs, max=clamp_abs)
    return True


def _save_log_vars(model, target_dir):
    log_vars = _get_log_vars_param(model)
    if log_vars is None:
        return
    ensure_dir(target_dir)
    torch.save(log_vars.detach().cpu(), os.path.join(target_dir, LOG_VARS_FILE))


def _load_log_vars(model, source_dir):
    log_vars = _get_log_vars_param(model)
    if log_vars is None:
        return False
    fp = os.path.join(source_dir, LOG_VARS_FILE)
    if not os.path.exists(fp):
        return False
    loaded = torch.load(fp, map_location=log_vars.device)
    if not torch.is_tensor(loaded):
        loaded = torch.tensor(loaded, dtype=log_vars.dtype, device=log_vars.device)
    loaded = loaded.to(device=log_vars.device, dtype=log_vars.dtype).view_as(log_vars)

    if not torch.isfinite(loaded).all():
        # Skip corrupted log_vars checkpoints and fall back to model init.
        return False

    with torch.no_grad():
        log_vars.copy_(loaded)
    return True


def _load_adapter_state_from_dir(model, ckpt_dir):
    safetensor_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    bin_path = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.exists(safetensor_path):
        set_peft_model_state_dict(model, safetensors_load_file(safetensor_path, device="cpu"))
        return safetensor_path
    if os.path.exists(bin_path):
        set_peft_model_state_dict(model, torch.load(bin_path, map_location="cpu"))
        return bin_path
    return None


class LossWeightsLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        try:
            import wandb

            self.wandb = wandb
        except Exception:
            self.wandb = None

    def on_step_end(self, args, state, control: TrainerControl, **kwargs):
        if args.logging_steps and state.global_step % args.logging_steps != 0:
            return control
        if not state.is_world_process_zero:
            return control
        if self.wandb is None:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        base_model = model.module if hasattr(model, "module") else model
        loss_dict = getattr(base_model, "last_loss_components", None)
        if not loss_dict:
            return control

        metrics = {f"loss_components/{k}": float(v) for k, v in loss_dict.items()}
        self.wandb.log(metrics, step=state.global_step)
        return control


class SaveLogVarsCallback(TrainerCallback):
    """Persist log_vars with every trainer checkpoint for consistent restore."""

    def on_save(self, args, state, control: TrainerControl, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        _save_log_vars(model, ckpt_dir)
        return control


class LogVarsSafetyCallback(TrainerCallback):
    """Apply post-step safety normalization for log_vars only when abnormal."""

    def __init__(self, clamp_abs=20.0):
        super().__init__()
        self.clamp_abs = float(clamp_abs)

    def on_train_begin(self, args, state, control: TrainerControl, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            _sanitize_log_vars(model, clamp_abs=self.clamp_abs)
        return control

    def on_optimizer_step(self, args, state, control: TrainerControl, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            _sanitize_log_vars(model, clamp_abs=self.clamp_abs)
        return control

    def on_step_end(self, args, state, control: TrainerControl, **kwargs):
        # Fallback path for older Trainer callback flows.
        model = kwargs.get("model", None)
        if model is not None:
            _sanitize_log_vars(model, clamp_abs=self.clamp_abs)
        return control


@torch.no_grad()
def build_token_frequency_weights(freq_dict, vocab_size, beta=0.997):
    weights = np.ones(vocab_size, dtype=np.float32)
    for idx, freq in freq_dict.items():
        if 0 <= idx < vocab_size:
            weights[idx] = (1.0 - beta) / (1.0 - beta ** freq)
    return torch.tensor(weights, dtype=torch.float32)


def _compute_gain_from_files(data_path, dataset, index_file, max_k=4):
    dataset_dir = os.path.join(data_path, dataset)
    index_path = os.path.join(dataset_dir, dataset + index_file)
    emb_path = os.path.join(dataset_dir, dataset + ".emb-llama-td.npy")
    if (not os.path.exists(index_path)) or (not os.path.exists(emb_path)):
        return None

    with open(index_path, "r") as f:
        data = json.load(f)
    if not data:
        return None

    embeddings = np.load(emb_path)
    if embeddings.ndim != 2:
        return None

    item_num, _ = embeddings.shape
    if item_num != len(data):
        return None

    token_lens = [len(tokens) for tokens in data.values() if isinstance(tokens, list)]
    if not token_lens:
        return None
    max_k = min(max_k, min(token_lens))
    if max_k <= 0:
        return None

    embed_mean = np.mean(embeddings, axis=0)
    layer0_avg_dist = float(np.linalg.norm(embeddings - embed_mean, axis=1).mean())

    weighted_means = []
    for k in range(1, max_k + 1):
        groups = defaultdict(list)
        for idx, tokens in data.items():
            if len(tokens) < k:
                continue
            groups[tuple(tokens[:k])].append(int(idx))
        if not groups:
            return None

        group_dists = []
        weights = []
        for indices in groups.values():
            group_embeds = embeddings[indices]
            group_mean = np.mean(group_embeds, axis=0)
            avg_dist = float(np.linalg.norm(group_embeds - group_mean, axis=1).mean())
            group_dists.append(avg_dist)
            weights.append(len(indices))

        group_dists = np.array(group_dists, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        weighted_means.append(float(np.sum(group_dists * weights) / np.sum(weights)))

    while len(weighted_means) < 4:
        weighted_means.append(0.0)

    gain = torch.tensor(
        [
            float(layer0_avg_dist - weighted_means[0]),
            float(weighted_means[0] - weighted_means[1]),
            float(weighted_means[1] - weighted_means[2]),
            float(weighted_means[2] - weighted_means[3]),
        ],
        dtype=torch.float32,
    )
    return gain


def load_gain(data_path, dataset, index_file):
    dynamic_gain = _compute_gain_from_files(data_path, dataset, index_file, max_k=4)
    if dynamic_gain is not None:
        print(f"[gain] using dynamically computed gain from dataset files: {dynamic_gain.tolist()}")
        return dynamic_gain

    gain = None
    if "Music" in dataset:
        if "pq" in index_file:
            gain = torch.tensor([7.220039215084789, 25.407310196792494, 7.437438597917831, 0.37421879562481], dtype=torch.float32)
        else:
            gain = torch.tensor([20.998990854202873, 9.038188096502836, 6.32627786428509, 4.012354740926474], dtype=torch.float32)
    elif "Industrial" in dataset:
        gain = torch.tensor([23.55882264845735, 11.152304234015027, 5.34162242456969, 0.7424608438967604], dtype=torch.float32)
    elif "Yelp" in dataset:
        if "pq" in index_file:
            gain = torch.tensor([3.733620935673425, 17.847017796673846, 4.796236244803253, 0.40488237758580314], dtype=torch.float32)
        else:
            gain = torch.tensor([8.580470237147342, 13.30601269284221, 4.458037499961477, 0.4284322915178977], dtype=torch.float32)
    elif "movie" in dataset:
        if "pq" in index_file:
            gain = torch.tensor([2.449707553892015, 16.96486981456009, 3.0170070197881262, 0.14662834979687772], dtype=torch.float32)
        else:
            gain = torch.tensor([10.019008538724833, 11.563622334615614, 0.9298545026668653, 0.06572736202979745], dtype=torch.float32)
    if gain is not None:
        print(f"[gain] using fallback hard-coded gain: {gain.tolist()}")
    else:
        print("[gain] unavailable; front_greater_loss will fall back to origin loss")
    return gain


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = {"": 0}
    print(vars(args))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    config = LlamaConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    print("add {} new token.".format(add_num))
    print("data num:", len(train_data))
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    with open(os.path.join(args.data_path, args.dataset, args.dataset + ".inter.json"), "r") as f:
        inters = json.load(f)
    with open(os.path.join(args.data_path, args.dataset, args.dataset + args.index_file), "r") as f:
        indices = json.load(f)

    freq = {}
    for _, inter in inters.items():
        training_inters = inter[:-2]
        if args.core7:
            training_inters = inter[:-4]
        for i in range(1, len(training_inters)):
            tokenized = tokenizer(
                "".join(indices[str(training_inters[i])]),
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )["input_ids"]
            for token_ids in tokenized:
                for token_id in token_ids:
                    tid = int(token_id)
                    freq[tid] = freq.get(tid, 0) + 1

    token_freq_weights = build_token_frequency_weights(freq, vocab_size=len(tokenizer))

    ig_dict = None
    if "igd" in args.method.lower():
        ig_comp = IGComputer()

        for _, inter in inters.items():
            for item in inter:
                ig_comp.item_freq[int(item)] += 1

        for item_id in indices:
            token_ids = tokenizer("".join(indices[str(item_id)]))["input_ids"]
            ig_comp.add_item(token_ids, int(item_id))

        ig_dict = ig_comp.compute_IG()

    gain = load_gain(args.data_path, args.dataset, args.index_file)

    collator = Collator(args, tokenizer)
    load_kwargs = dict(
        device_map=device_map,
        low_cpu_mem_usage=True,
        args=args,
        item_weights=None,
        token_weights=token_freq_weights,
        gain=gain,
        igd=ig_dict,
    )
    if args.load_in_8bit:
        model = LETTER.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            **load_kwargs,
        )
    else:
        if args.bf16:
            load_dtype = torch.bfloat16
        elif args.fp16:
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32
        model = LETTER.from_pretrained(
            args.base_model,
            torch_dtype=load_dtype,
            **load_kwargs,
        )
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))

    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    if "ours" in args.method.lower():
        _ensure_log_vars_trainable(model)

    if args.resume_from_checkpoint:
        resume_dir = args.resume_from_checkpoint
        args.resume_from_checkpoint = False
        loaded_from = _load_adapter_state_from_dir(model, resume_dir)
        if loaded_from:
            print(f"Restarting adapter from {loaded_from}")
            loaded_log_vars = _load_log_vars(model, resume_dir)
            if "ours" in args.method.lower():
                print(f"Loaded {LOG_VARS_FILE}: {loaded_log_vars}")
        else:
            print(f"Checkpoint not found in {resume_dir} (adapter_model.safetensors / adapter_model.bin)")

    for n, p in model.named_parameters():
        if "original_module" in n and any(module_name in n for module_name in peft_config.modules_to_save):
            p.requires_grad = False
    if "ours" in args.method.lower():
        _ensure_log_vars_trainable(model)

    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    callbacks = (
        [LossWeightsLoggerCallback(), SaveLogVarsCallback(), LogVarsSafetyCallback(clamp_abs=20.0)]
        if "ours" in args.method.lower()
        else []
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            report_to=["wandb"],
            run_name=f"{args.method}_{args.dataset}_seed{args.seed}",
            fp16=args.fp16,
            bf16=args.bf16,
            tf32=args.tf32,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=args.gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            deepspeed=args.deepspeed,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=(args.dataloader_num_workers > 0),
            group_by_length=args.group_by_length,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )
    model.config.use_cache = False

    if args.use_torch_compile and torch.__version__ >= "2" and sys.platform != "win32":
        trainer.model = torch.compile(trainer.model)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if "ours" in args.method.lower():
        # Trainer may reload the best adapter weights at end; align log_vars to the same checkpoint.
        best_ckpt = trainer.state.best_model_checkpoint
        if best_ckpt:
            loaded_log_vars = _load_log_vars(trainer.model, best_ckpt)
            print(f"Best checkpoint: {best_ckpt}")
            print(f"Loaded {LOG_VARS_FILE} from best checkpoint: {loaded_log_vars}")
        _ensure_log_vars_trainable(trainer.model)

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
    if "ours" in args.method.lower():
        _save_log_vars(trainer.model, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()
    train(args)
