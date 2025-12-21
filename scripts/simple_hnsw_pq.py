import numpy as np
import faiss
import json
import time
from compute_wiki_recall import compute_recall_against_gt

emb = np.load("wiki_subset.npy")
query = np.load("wiki_query.npy")

# print("prepare criterion")

# Retrieve HNSW stats
# index_key = "HNSW64,PQ32"
index_key = "HNSW64"
# index_key = "IVF4096,Flat"
index = faiss.index_factory(768, index_key)

# index.train(xt)
# index.add(xb)
index.train(emb)
index.add(emb)

# index.hnsw.prune_state = 0
# index.add(emb)

print("done add")

# k=gt.shape[1]

t0 = time.time()
D, I = index.search(query, 10) # sanity check
latency = time.time() - t0
# print(I)
# print(D)

print(f"{latency:.3f}")

# results_list = I.tolist()
# recall = compute_recall_against_gt(results_list)
# print(f"{latency:.3f}, {recall:.3f}")


# with open("ivf_wiki.json", "w") as f:
#     json.dump(results_list, f)