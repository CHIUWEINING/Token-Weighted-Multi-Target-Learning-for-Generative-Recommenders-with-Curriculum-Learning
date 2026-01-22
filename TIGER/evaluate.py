import math
import re
import numpy as np
from collections import defaultdict
def get_topk_results(predictions, scores, targets, other_targets, k, output_groups, item_groups, all_items=None, args=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions)##################
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        if output_groups:
            batch_seqs = []
            batch_scores = []
            for i in range(b * k, (b + 1) * k):
                batch_seqs.append(predictions[i])
                batch_scores.append(scores[i])
                for pair in output_groups[i]:
                    batch_seqs.append(pair[0])
                    batch_scores.append(pair[1])
                if len(batch_seqs) >= k:
                    break
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_items = [targets[b]]
        if args.core7:
            target_items += [other_target for other_target in other_targets[b]]
        one_results = defaultdict(list)
        count = 0
        for sorted_pred in sorted_pairs:
            if(count >= k):break
            compare1_4 = sorted_pred[0]
            for target_item in target_items:
                compare2_4 = target_item
                if args.group_match:
                    compare1_3 = re.sub(r'<[^>]*>$', '', sorted_pred[0])#3
                    compare2_3 = re.sub(r'<[^>]*>$', '', target_item)#3
                    
                    compare1_2 = re.sub(r'((?:<[^>]*>){2}).*', r'\1', sorted_pred[0]) #2
                    compare2_2 = re.sub(r'((?:<[^>]*>){2}).*', r'\1', target_item)#2
                    
                    compare1_1 = re.sub(r'(<[^>]*>).*$' , r'\1', sorted_pred[0])#1
                    compare2_1 = re.sub(r'(<[^>]*>).*$' , r'\1', target_item)#1
                    if len(one_results["1"]) == count and compare1_1 == compare2_1 :
                        one_results["1"].append(1)
                    if len(one_results["2"]) == count and compare1_2 == compare2_2:
                        one_results["2"].append(1)
                    if len(one_results["3"]) == count and compare1_3 == compare2_3:
                        one_results["3"].append(1)

                if len(one_results["4"]) == count and compare1_4 == compare2_4:# sorted_pred[0] == target_item
                    one_results["4"].append(1)
            if len(one_results["4"]) == count:
                one_results["4"].append(0)
            if args.group_match:
                if len(one_results["1"]) == count: one_results["1"].append(0)
                if len(one_results["2"]) == count: one_results["2"].append(0)
                if len(one_results["3"]) == count: one_results["3"].append(0)
            count+=1
            if(count >= k):break
            
            

        results.append(one_results)

    return results

def partial_correct(predictions, scores, targets, k, all_items=None, args=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions)##################
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        count = 0
        correct_tokens = []
        wrong_tokens = []
        for sorted_pred in sorted_pairs:
            if(count >= k):break
            pred =  re.findall(r"<[^>]+>", sorted_pred[0])
            target_token = re.findall(r"<[^>]+>", target_item)
            correct_count = 0
            for it in target_token:
                if it in pred:
                    correct_count += 1
                    correct_tokens.append(it)
                else:
                    wrong_tokens.append(it)
            if correct_count > 0:
                one_results.append(1)
            else:
                one_results.append(0)
            
            count+=1
            

        results.append(one_results)

    return results, correct_tokens, wrong_tokens

def get_topk_distance_results(predictions, scores, targets, k, target_ids, id2num, embs, args=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions)##################
    dist = []
    for b in range(B):
        target_item = targets[b]
        target_num = id2num[target_item]
        target_emb = embs[target_num]
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        batch_weight = batch_scores / batch_scores.sum()
        
        batch_dist = 0.0
        for i, item in enumerate(batch_seqs):
            if item in id2num:
                num = id2num[item]
            else:
                raise ValueError("item not in id2num")
            
            emb1 = embs[num]
            cos_sim = np.dot(emb1, target_emb) / (np.linalg.norm(emb1) * np.linalg.norm(target_emb)) 
            batch_dist += cos_sim * batch_weight[i]
        
        dist.append(batch_dist.item())
    return dist
            
def get_predictions(predictions, scores, id2num, targets, k):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    for b in range(B):
        result = {}
        target_item = targets[b]
        target_num = id2num[target_item]
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        result["target_id"] = target_num
        result["target"] = target_item
        result["predictions_id"] = [id2num[pair[0]] for pair in pairs]
        result["predictions"] = [pair[0] for pair in pairs]
        results.append(result)
        
    return results

def get_topk_ranking_results(predictions, targets, k, all_items=None):
    results = []
    B = len(targets)

    for b in range(B):
        batch_seqs = predictions[b]
        target_item = targets[b]
        one_results = []
        for sorted_pred in predictions:
            if sorted_pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results
def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        for i in range(4, 0, -1):
            if str(i) in topk_results[0]:
                if m.lower().startswith("hit"):
                    k = int(m.split("@")[1])
                    res[m+f"_{i}"] = hit_k(topk_results, k, str(i))
                elif m.lower().startswith("ndcg"):
                    k = int(m.split("@")[1])
                    res[m+f"_{i}"] = ndcg_k(topk_results, k, str(i))
                else:
                    raise NotImplementedError

    return res


def ndcg_k(topk_results, k, match_key):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[match_key][:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k, match_key):
    hit = 0.0
    for row in topk_results:
        res = row[match_key][:k]
        if sum(res) > 0:
            hit += 1
    return hit


import re

def _normalize_preds(preds):
    return [p.strip().replace(" ", "") for p in preds]

def _batch_topk_pairs(preds, scores, b, k, output_groups=None):
    """Return top-k (item, score) pairs for batch b."""
    start, end = b * k, (b + 1) * k
    batch_seqs = preds[start:end]
    batch_scores = scores[start:end]

    if output_groups:
        # expand with group candidates (pred, score) appended after each main pred
        expanded_seqs, expanded_scores = [], []
        for i in range(start, end):
            expanded_seqs.append(preds[i])
            expanded_scores.append(scores[i])
            for pair in output_groups[i]:
                expanded_seqs.append(pair[0])
                expanded_scores.append(pair[1])
                if len(expanded_seqs) >= k:
                    break
        batch_seqs, batch_scores = expanded_seqs, expanded_scores

    pairs = list(zip(batch_seqs, batch_scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]

def _matches_target(pred_item, target_item, group_match=False):
    """True if pred_item counts as the target (with your group rules)."""
    a = re.sub(r'<[^>]*>$', '', pred_item)  if group_match else pred_item
    b = re.sub(r'<[^>]*>$', '', target_item) if group_match else target_item
    if a == b:
        return True
    return False

def _filter_invalid_with_all_items(preds, scores, all_items):
    if all_items is None:
        return scores
    # mutate a copy so we don't touch the original list
    new_scores = list(scores)
    for i, seq in enumerate(preds):
        if seq not in all_items:
            new_scores[i] = -1000
    return new_scores

def find_fail1_success2_cases( step,
    predictions1, scores1,
    predictions2, scores2,
    targets, k=10,
    output_groups=None,          # list (same length as predictions*) of lists[(pred,score)] or None            # dict: base_item -> list[variants]
    all_items=None,              # set/list of allowed items
    args=None                    # expects .group_match (bool) if provided
):
    """
    Returns a list of dicts for batches b where:
      - target NOT in top-k of (predictions1,scores1)
      - target IS in top-k of (predictions2,scores2)

    Each dict contains:
      {
        'index': b,
        'target': targets[b],
        'topk1': [(item, score), ...],
        'topk2': [(item, score), ...]
      }
    """
    group_match = getattr(args, "group_match", False) if args is not None else False

    # Normalize predictions (remove spaces like your original)
    predictions1 = _normalize_preds(predictions1)
    predictions2 = _normalize_preds(predictions2)

    # Respect all_items filtering (mirror your earlier logic)
    scores1 = _filter_invalid_with_all_items(predictions1, scores1, all_items)
    scores2 = _filter_invalid_with_all_items(predictions2, scores2, all_items)

    B = len(targets)
    results = []

    for b in range(B):
        topk1 = _batch_topk_pairs(predictions1, scores1, b, k, output_groups)
        topk2 = _batch_topk_pairs(predictions2, scores2, b, k, output_groups)

        target_item = targets[b]
        results1 = [_matches_target(p, target_item, group_match) for p, _ in topk1]
        results2 = [_matches_target(p, target_item, group_match) for p, _ in topk2]
        hit1 = any(results1)
        hit2 = any(results2)

        if (not hit1) and hit2:
            for i in range(len(results2)):
                if results2[i]:
                    hit_id = i
                    break
            results.append({
                "index": step*B+b,
                "target": target_item,
                "hit_id": hit_id,
                "topk1": topk1,   # first set's top-k (missed)
                "topk2": topk2,   # second set's top-k (hit)
            })

    return results

