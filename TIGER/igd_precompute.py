# igd_precompute.py
import json
import math
from collections import defaultdict

class IGComputer:
    def __init__(self):
        self.prefix_trie = {}                      # node → dict(subtoken → child)
        self.prefix_items = defaultdict(set)       # prefix_tuple → {item_ids}
        self.item_freq = defaultdict(int)          # prior p(i)
        self.entropy_cache = {}                    # prefix_tuple → H
        self.IG_dict = {}                          # (prefix_tuple, token) → IG

    def add_item(self, token_ids, item_id):
        # token_ids is a list of token ids
        prefix = []
        for t in token_ids:
            self.prefix_items[tuple(prefix)].add(item_id)
            prefix.append(t)
        self.prefix_items[tuple(prefix)].add(item_id)

    def compute_entropy(self, prefix_tuple):
        if prefix_tuple in self.entropy_cache:
            return self.entropy_cache[prefix_tuple]

        items = self.prefix_items[prefix_tuple]
        total = sum(self.item_freq[i] for i in items)
        H = 0.0
        for i in items:
            p = self.item_freq[i] / total if total != 0 else 1 / len(items)
            H -= p * math.log(p + 1e-12)
        self.entropy_cache[prefix_tuple] = H
        return H

    def compute_IG(self):
        for prefix, items in self.prefix_items.items():
            H_prev = self.compute_entropy(prefix)

            for token in set(k[len(prefix)] for k in self.prefix_items if len(k) > len(prefix) and k[:len(prefix)] == prefix):
                new_prefix = prefix + (token,)
                H_new = self.compute_entropy(new_prefix)
                IG = H_prev - H_new
                self.IG_dict[(prefix, token)] = max(IG, 0.0)  # ensure non-negative

        return self.IG_dict
