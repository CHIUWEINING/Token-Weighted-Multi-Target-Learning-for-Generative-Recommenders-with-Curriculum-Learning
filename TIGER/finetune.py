import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
from typing import List
from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling_letter import LETTER
# import wandb
from utils import *
from collator import Collator
from transformers import TrainerCallback, TrainerControl
from igd_precompute import IGComputer

class LossWeightsLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        try:
            import wandb
            self.wandb = wandb
        except Exception:
            self.wandb = None

    def on_step_end(self, args, state, control: TrainerControl, **kwargs):
        # respect logging cadence (only meaningful when logging_strategy="steps")
        if args.logging_steps and state.global_step % args.logging_steps != 0:
            return control
        # only main process logs in DDP
        if not state.is_world_process_zero:
            return control
        if self.wandb is None:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control
        d = getattr(model, "last_loss_components", None)
        if not d:
            return control

        metrics = {f"loss_components/{k}": float(v) for k, v in d.items()}
        # log directly to W&B with the step
        self.wandb.log(metrics, step=state.global_step)
        return control


def train(args):
    beta = 0.99
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)


    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    args.deepspeed = None
    gradient_checkpointing= False


    train_data, valid_data = load_datasets(args)

    with open(os.path.join(args.data_path, args.dataset, args.dataset + ".inter.json"), 'r') as f:
        inters = json.load(f)

    with open(os.path.join(args.data_path, args.dataset, args.dataset + args.index_file), 'r') as f:
        indices = json.load(f)
            
    

    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[0])
        print(valid_data[0])

    freq = dict()
    for uid, inter in inters.items():
        training_inters = inter[:-2]
        if args.core7:
            training_inters = inter[:-4]
        for i in range(1, len(training_inters)):
            inputs = tokenizer("".join(indices[str(training_inters[i])]),
                               return_tensors="pt",
                                padding="longest",
                                max_length=tokenizer.model_max_length,
                                truncation=True)
            inputs = inputs["input_ids"]
            for token_ids in inputs:
                for token_id in token_ids:
                    token_id = int(token_id)
                    if token_id not in freq:
                        freq[token_id] = 0
                    freq[token_id] += 1

    @torch.no_grad()
    def build_token_frequency_weights(freq_dict, beta = 0.9):
        import numpy as np, torch
        num_tokens = max(freq_dict) + 1         
        weights = np.ones(num_tokens, dtype=np.float32)
        for idx, f in freq_dict.items():
            # print((1. - beta) / (1. - beta ** f), f)
            weights[idx] = (1. - beta) / (1. - beta ** f)
        weights = torch.tensor(weights,  dtype=torch.float32)
        return weights #/ weights.mean()         # stablize gradient norm


    IG_dict = None
    if "igd" in args.method.lower():
        ig_comp = IGComputer()

        # build item prior freq
        for uid, inter in inters.items():
            for item in inter:
                ig_comp.item_freq[int(item)] += 1

        # build prefix → item set
        for item_id in indices:
            token_ids = tokenizer("".join(indices[str(item_id)]))["input_ids"]
            ig_comp.add_item(token_ids, int(item_id))

        IG_dict = ig_comp.compute_IG()
    token_freq_weights = build_token_frequency_weights(freq)
    
    gain = None
    ##Calculated from codebook_dist.py
    if "Music" in args.dataset:
        if "pq" in args.index_file:
            gain = torch.tensor([7.220039215084789, 25.407310196792494, 7.437438597917831, 0.37421879562481], dtype=torch.float32)
        else:
            gain = torch.tensor([20.998990854202873, 9.038188096502836, 6.32627786428509, 4.012354740926474], dtype=torch.float32)
    elif "Industrial" in args.dataset:
        gain = torch.tensor([23.55882264845735, 11.152304234015027, 5.34162242456969, 0.7424608438967604], dtype=torch.float32)
    elif "Yelp" in args.dataset:
        if "pq" in args.index_file:
            gain = torch.tensor([3.733620935673425, 17.847017796673846, 4.796236244803253, 0.40488237758580314], dtype=torch.float32)
        else:
            gain = torch.tensor([8.580470237147342, 13.30601269284221, 4.458037499961477, 0.4284322915178977], dtype=torch.float32)
    elif "movie" in args.dataset:
        if "pq" in args.index_file:
            gain = torch.tensor([2.449707553892015, 16.96486981456009, 3.0170070197881262, 0.14662834979687772], dtype=torch.float32)
        else:
            gain = torch.tensor([10.019008538724833, 11.563622334615614, 0.9298545026668653, 0.06572736202979745], dtype=torch.float32)
    
    
    collator = Collator(args, tokenizer)
    model = LETTER(config, item_weights=None, token_weights=token_freq_weights, gain = gain, args=args, igd = IG_dict)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if local_rank == 0:
        print(model)

    print(max(token_freq_weights), min(token_freq_weights), token_freq_weights.shape)
    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True
    callbacks = [EarlyStoppingCallback(early_stopping_patience=20),\
                LossWeightsLoggerCallback()] if "ours" in args.method  \
                else [EarlyStoppingCallback(early_stopping_patience=20)]

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
            # fp16=args.fp16,
            # bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            # gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            # deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=['wandb'],
            run_name=f"{args.method}_{args.dataset}_seed{args.seed}",
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks = callbacks,
    )
    model.config.use_cache = False


    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    args.output_dir += args.dataset + '/'
    train(args)
