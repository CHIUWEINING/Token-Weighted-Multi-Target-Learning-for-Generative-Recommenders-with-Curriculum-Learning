import math
import re
import numpy as np
from collections import defaultdict


def _normalize(predictions):
    normalized = []
    for p in predictions:
        p = p.split("Response:")[-1]
        normalized.append(p.strip().replace(" ", ""))
    return normalized


def get_topk_results(predictions, scores, targets, other_targets, k, output_groups, item_groups, all_items=None, args=None):
    results = []
    bsz = len(targets)
    predictions = _normalize(predictions)

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(bsz):
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

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
        if args is not None and args.core7:
            target_items += [other_target for other_target in other_targets[b]]

        one_results = defaultdict(list)
        count = 0
        for sorted_pred in sorted_pairs:
            if count >= k:
                break

            compare1_4 = sorted_pred[0]
            for target_item in target_items:
                compare2_4 = target_item
                if args is not None and args.group_match:
                    compare1_3 = re.sub(r"<[^>]*>$", "", sorted_pred[0])
                    compare2_3 = re.sub(r"<[^>]*>$", "", target_item)

                    compare1_2 = re.sub(r"((?:<[^>]*>){2}).*", r"\1", sorted_pred[0])
                    compare2_2 = re.sub(r"((?:<[^>]*>){2}).*", r"\1", target_item)

                    compare1_1 = re.sub(r"(<[^>]*>).*$", r"\1", sorted_pred[0])
                    compare2_1 = re.sub(r"(<[^>]*>).*$", r"\1", target_item)

                    if len(one_results["1"]) == count and compare1_1 == compare2_1:
                        one_results["1"].append(1)
                    if len(one_results["2"]) == count and compare1_2 == compare2_2:
                        one_results["2"].append(1)
                    if len(one_results["3"]) == count and compare1_3 == compare2_3:
                        one_results["3"].append(1)

                if len(one_results["4"]) == count and compare1_4 == compare2_4:
                    one_results["4"].append(1)

            if len(one_results["4"]) == count:
                one_results["4"].append(0)

            if args is not None and args.group_match:
                if len(one_results["1"]) == count:
                    one_results["1"].append(0)
                if len(one_results["2"]) == count:
                    one_results["2"].append(0)
                if len(one_results["3"]) == count:
                    one_results["3"].append(0)

            count += 1
            if count >= k:
                break

        results.append(one_results)

    return results


def partial_correct(predictions, scores, targets, k, all_items=None, args=None):
    results = []
    bsz = len(targets)
    predictions = _normalize(predictions)

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(bsz):
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        target_item = targets[b]
        one_results = []
        correct_tokens = []
        wrong_tokens = []

        for sorted_pred in sorted_pairs:
            pred = re.findall(r"<[^>]+>", sorted_pred[0])
            target_token = re.findall(r"<[^>]+>", target_item)
            correct_count = 0
            for token in target_token:
                if token in pred:
                    correct_count += 1
                    correct_tokens.append(token)
                else:
                    wrong_tokens.append(token)
            one_results.append(1 if correct_count > 0 else 0)

        results.append(one_results)

    return results, correct_tokens, wrong_tokens


def get_topk_distance_results(predictions, scores, targets, k, target_ids, id2num, embs, args=None):
    predictions = _normalize(predictions)
    dist = []

    for b in range(len(targets)):
        target_item = targets[b]
        target_num = id2num[target_item]
        target_emb = embs[target_num]

        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

        batch_weight = batch_scores / batch_scores.sum()
        batch_dist = 0.0

        for i, item in enumerate(batch_seqs):
            if item not in id2num:
                continue
            num = id2num[item]
            emb1 = embs[num]
            cos_sim = np.dot(emb1, target_emb) / (np.linalg.norm(emb1) * np.linalg.norm(target_emb))
            batch_dist += cos_sim * batch_weight[i]

        dist.append(batch_dist.item() if hasattr(batch_dist, "item") else float(batch_dist))

    return dist


def get_predictions(predictions, scores, id2num, targets, k):
    results = []
    predictions = _normalize(predictions)

    for b in range(len(targets)):
        result = {}
        target_item = targets[b]
        target_num = id2num.get(target_item, -1)

        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]

        result["target_id"] = target_num
        result["target"] = target_item
        result["predictions_id"] = [id2num.get(pair[0], -1) for pair in pairs]
        result["predictions"] = [pair[0] for pair in pairs]
        results.append(result)

    return results


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        for i in range(4, 0, -1):
            if str(i) in topk_results[0]:
                if m.lower().startswith("hit"):
                    k = int(m.split("@")[1])
                    res[m + f"_{i}"] = hit_k(topk_results, k, str(i))
                elif m.lower().startswith("ndcg"):
                    k = int(m.split("@")[1])
                    res[m + f"_{i}"] = ndcg_k(topk_results, k, str(i))
                else:
                    raise NotImplementedError
    return res


def ndcg_k(topk_results, k, match_key):
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
