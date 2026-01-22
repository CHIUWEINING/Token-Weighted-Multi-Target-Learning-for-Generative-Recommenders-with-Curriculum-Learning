import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from typing import List

import torch
import transformers
# from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results, \
                    get_topk_distance_results, get_predictions
from generation_trie import Trie


import torch.nn.functional as F
import re

def seq_logprob(model, tokenizer, encoder_inputs, item_text, device):
    """回傳 log P(item_text | encoder_inputs)"""
    # 1) tokenise item → labels
    labels = tokenizer(item_text, return_tensors="pt").input_ids.to(device)

    # 2) T5 需要 shift_right 作為 decoder_input_ids
    decoder_input_ids = model._shift_right(labels)

    # 3) 前向；拿 logits
    with torch.no_grad():
        out = model(
            input_ids      = encoder_inputs["input_ids"].unsqueeze(0),   # [1, L_in]
            attention_mask = encoder_inputs["attention_mask"].unsqueeze(0),
            decoder_input_ids = decoder_input_ids
        )
        logits = out.logits                         # [1, L_out, |V|]
        log_probs = F.log_softmax(logits, dim=-1)   # 同 shape

        # 4) gather 正確 token 的 log P，並沿時間相加
        seq_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1).sum()  # 標量
    return seq_lp 
def test(args):

    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda",args.gpu_id)

    config = T5Config.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small",
        model_max_length=512,
    )
    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    
    
    config.vocab_size = len(tokenizer)

    print("add {} new token.".format(add_num))
    print("data num:", len(train_data))

    # tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    prompt_ids = [0]

    test_data = load_test_dataset(args)
    add_num += tokenizer.add_tokens(test_data.get_new_tokens())
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()
    
    
    item_groups = {}
            

    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=4, pin_memory=True)

    raw_users, raw_labels = test_data.process_raw_labels()
    print("data num:", len(test_data))  
    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []

    id2num = dict()
    for num, index in test_data.indices.items():
        id2num["".join(index)] = int(num)

    if args.dist_eval:
        embs = np.load(f"../data/{args.dataset}/{args.dataset}.emb-llama-td.npy", allow_pickle=True)
        total_dist = []
    total_predictions = []
    total_output = {"origin":dict(), "index":dict()}
        
    with torch.no_grad():
        for prompt_id in prompt_ids:

            
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                target_ids = batch[2]
                other_targets = batch[3]
                per_batch = len(targets)
                # raw_tagrets = raw_labels[total:total+per_batch]
                raw_current_users = raw_users[total:total+per_batch]
                total += per_batch
                if step == 0:
                    print(inputs)
                    print(targets)

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    # max_length=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                # print(output_ids.shape)
                # print(np.array(output).shape)
                # print(type((output[0])),output[0])
                output_groups = []
                
                if args.output_predictions:
                    predictions = get_predictions(output, scores, id2num, targets, args.num_beams)
                    total_predictions.extend(predictions)
                    for i in range(len(predictions)):
                        total_output["origin"][raw_current_users[i]] = [str(it) for it in predictions[i]["predictions_id"]]
                        total_output["index"][raw_current_users[i]] = predictions[i]["predictions"]
                    
                topk_res = get_topk_results(output, scores, targets, other_targets, args.num_beams, output_groups, item_groups,
                                            all_items=all_items if args.filter_items else None, args=args)
                
                    
                if args.dist_eval:
                    topk_dist = get_topk_distance_results(output, scores, targets, args.num_beams, target_ids, id2num, embs, args=args)
                    total_dist += topk_dist
                    # print(total_dist)
                    print("Current Avg Distance: ", np.array(total_dist).sum()/len(total_dist))
                batch_metrics_res = get_metrics_results(topk_res, metrics)
                # print(batch_metrics_res)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                # if (step+1)%10 == 0:
                temp={}
                for m in metrics_results:
                    temp[m] = metrics_results[m] / total
                print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in all_prompt_results[0]:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")


    save_data={}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results
    if args.dist_eval:
        save_data["avg distance score"] = np.array(total_dist).sum()/len(total_dist)
        print("Final Avg Distance: ", np.array(total_dist).sum()/len(total_dist))
    if args.group_match:
        args.results_file = args.results_file.replace(".json", "_groupmatch.json")
    # if args.core7:
    #     args.results_file = args.results_file.replace(".json", "_core7.json")
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)

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
            # Use 0 to make KCW life easier
            # Normal LETTER-TIGER only has 0 as well.
            json.dump({"origin": total_output["origin"],
                    "index": total_output["index"]}, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)
