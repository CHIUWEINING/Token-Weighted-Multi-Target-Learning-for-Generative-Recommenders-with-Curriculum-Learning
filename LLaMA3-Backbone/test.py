import argparse
import json
import os

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import (
    get_metrics_results,
    get_predictions,
    get_topk_distance_results,
    get_topk_results,
)


def test(args):
    set_seed(args.seed)
    print(vars(args))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device("cuda", args.gpu_id)
    device_map = {"": args.gpu_id}

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if args.lora:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
        else:
            prompt_ids = [0]
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()
    output_groups = []
    item_groups = {}

    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("data num:", len(test_data))

    id2num = {"".join(index): int(num) for num, index in test_data.indices.items()}
    embs = None
    if args.dist_eval:
        embs = np.load(f"../data/{args.dataset}/{args.dataset}.emb-llama-td.npy", allow_pickle=True)

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    total_predictions = []
    total_output = {"origin": {}, "index": {}}
    total_dist = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            print("Start prompt:", prompt_id)

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                target_ids = batch[2]
                other_targets = batch[3]
                users = batch[4]

                total += len(targets)
                num_beams = args.num_beams
                while True:
                    try:
                        output = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            return_legacy_cache=True,
                            early_stopping=True,
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        print("Out of memory!")
                        num_beams = max(num_beams - 1, 1)
                        print("Beam:", num_beams)
                    except Exception:
                        raise RuntimeError

                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                topk_res = get_topk_results(
                    decoded,
                    scores,
                    targets,
                    other_targets,
                    num_beams,
                    output_groups,
                    item_groups,
                    all_items=all_items if args.filter_items else None,
                    args=args,
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if args.dist_eval:
                    topk_dist = get_topk_distance_results(decoded, scores, targets, num_beams, target_ids, id2num, embs, args=args)
                    total_dist += topk_dist

                if args.output_predictions:
                    pred_records = get_predictions(decoded, scores, id2num, targets, num_beams)
                    total_predictions.extend(pred_records)
                    for i, record in enumerate(pred_records):
                        if users[i] is None:
                            continue
                        total_output["origin"][users[i]] = [str(it) for it in record["predictions_id"]]
                        total_output["index"][users[i]] = record["predictions"]

                if (step + 1) % 50 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / max(total, 1)
                    print(temp)
                    if args.dist_eval and len(total_dist) > 0:
                        print("Current Avg Distance:", float(np.array(total_dist).sum() / len(total_dist)))

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / max(total, 1)

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    if all_prompt_results:
        for m in all_prompt_results[0]:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")

    save_data = {}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    if args.dist_eval and len(total_dist) > 0:
        save_data["avg distance score"] = float(np.array(total_dist).sum() / len(total_dist))
        print("Final Avg Distance:", save_data["avg distance score"])

    if args.group_match:
        args.results_file = args.results_file.replace(".json", "_groupmatch.json")

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)
    print("Save file:", args.results_file)

    if args.output_predictions:

        def tensor_to_python(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        output_path = args.results_file.replace(".json", "_predictions.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in total_predictions:
                f.write(json.dumps(record, ensure_ascii=False, default=tensor_to_python) + "\n")

        output_path = args.results_file.replace(".json", "_predictions.json")
        with open(output_path, "w") as f:
            json.dump({"origin": total_output["origin"], "index": total_output["index"]}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)
