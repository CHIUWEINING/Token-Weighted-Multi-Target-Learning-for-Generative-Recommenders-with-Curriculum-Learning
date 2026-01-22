import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_topk_layer_distances(json_path, npy_path, dataset_name, max_k=4):
    # Load JSON and embeddings
    with open(json_path, "r") as f:
        data = json.load(f)
    embeddings = np.load(npy_path)
    item_num, embed_dim = embeddings.shape
    embed_mean = np.mean(embeddings, axis=0)
    dists = np.linalg.norm(embeddings - embed_mean, axis=1)  # Euclidean
    layer0_avg_dist = np.mean(dists)
    assert item_num == len(data), f"Mismatch: {item_num} embeddings vs {len(data)} JSON entries"

    all_group_distances = []  # stores list of per-group avg distances for each k
    weighted_means = []

    # ----- Loop over top-k levels -----
    for k in range(1, max_k + 1):
        groups = defaultdict(list)

        # Group items by their top-k tokens
        for idx, tokens in data.items():
            idx = int(idx)
            key = tuple(tokens[:k])  # top-k tuple as the group key
            groups[key].append(idx)

        # Compute avg distance within each group
        group_dists = []
        weights = []
        for key, indices in groups.items():
            group_embeds = embeddings[indices]
            group_mean = np.mean(group_embeds, axis=0)
            dists = np.linalg.norm(group_embeds - group_mean, axis=1)  # Euclidean
            avg_dist = np.mean(dists)
            group_dists.append(avg_dist)
            weights.append(len(indices))

        group_dists = np.array(group_dists)
        weights = np.array(weights)

        weighted_mean = np.sum(group_dists * weights) / np.sum(weights)
        all_group_distances.append(group_dists)
        weighted_means.append(weighted_mean)

    # ----- Plot -----
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_group_distances,
                labels=[f"Top-{k}" for k in range(1, max_k + 1)],
                showmeans=False,
                meanline=False,
                showfliers=False)
    # Overlay your weighted mean as horizontal lines
    for i, wm in enumerate(weighted_means, start=1):
        plt.hlines(
            wm,                       # y position = weighted mean
            i - 0.22, i + 0.22,       # x-range: a short line centered on box
            colors='red',
            linestyles='--',
            linewidth=2,
            label='Weighted mean' if i == 1 else None
        )
    plt.title(f"{dataset_name} - Top-k Token Group Distance")
    plt.xlabel("Top-k Tokens Used for Grouping")
    plt.ylabel("Average Euclidean Distance")
    plt.grid(True, linestyle='--', alpha=0.5)

    save_name = f"./{dataset_name}/{dataset_name}{'_pq' if 'pq' in json_path else ''}_topk_token_distance_boxplot.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ Saved box plot as '{save_name}'")

    # ----- Print Summary -----
    print("Layer 0: ", layer0_avg_dist)
    for k, mean in enumerate(weighted_means, start=1):
        print(f"Top-{k} weighted mean distance: {mean:.4f}")
    print([float(layer0_avg_dist - weighted_means[0]), float(weighted_means[0] - weighted_means[1]), \
            float(weighted_means[1] - weighted_means[2]), float(weighted_means[2] - weighted_means[3])])
    return weighted_means, all_group_distances


# Example usage:
dataset = "Random_Hashing_Musical_Instruments" #Yelp, movieLens
pq="" # empty string for rq, "_pq" for pq
compute_topk_layer_distances(f"../data/{dataset}/{dataset}{pq}.index.json",\
                        f"../data/{dataset}/{dataset}.emb-llama-td.npy", \
                        dataset, max_k=4)