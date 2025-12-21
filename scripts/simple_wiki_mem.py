import numpy as np
import faiss
import json
import time
# from compute_wiki_recall import compute_recall_json

emb = np.load("wiki_subset.npy")
query = np.load("wiki_query.npy")

# print("prepare criterion")

# Retrieve HNSW stats
index_key = "HNSW64,PQ32"
# index_key = "HNSW64"
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

# t0 = time.time()
# D, I = index.search(query, 10) # sanity check
# latency = time.time() - t0
# print(f"{latency:.3f}")
# # print(I)
# # print(D)

# results_list = I.tolist()



# with open("ivf_wiki.json", "w") as f:
#     json.dump(results_list, f)

def get_hnsw_pq_memory_breakdown(index):
    """
    Returns the memory usage of the Graph (Node) and Storage (Index) separately.
    Assumes index is an instance of IndexHNSWPQ.
    """
    # 1. Calculate Total Size
    # Serializing the whole index gives the total persistent size
    total_size = len(faiss.serialize_index(index))
    
    # 2. Calculate Index Memory (Vector Storage)
    # The 'storage' attribute in IndexHNSWPQ is the underlying IndexPQ
    storage_index = index.storage
    index_memory = len(faiss.serialize_index(storage_index))
    
    # 3. Calculate Node Memory (Graph)
    # The difference roughly equals the graph structure overhead
    # (Links, levels, and HNSW metadata)
    node_memory = total_size - index_memory
    

    print(f"Total Memory (MB): {total_size / (1024**2)}")
    print(f"Index/Vector Memory (MB): {index_memory / (1024**2)}")
    print(f"Node/Graph Memory (MB): {node_memory / (1024**2)}")
    

size_in_bytes = get_hnsw_pq_memory_breakdown(index)
print(f"Memory Usage: {size_in_bytes / (1024**2):.2f} MB")

# chunk = faiss.serialize_index(index)
# print(f"Index size: {len(chunk) / (1024**2):.2f} MB")


