import numpy as np
import faiss
import time

from faiss import class_wrappers
# Patch OneRecallAtRCriterion's evaluate method to convert np arrays to float*
class_wrappers.handle_AutoTuneCriterion(faiss.IntersectionCriterion)

def ivecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

print("load data")

xt = fvecs_read("../sift1M/sift_learn.fvecs")
xb = fvecs_read("../sift1M/sift_base.fvecs")
xq = fvecs_read("../sift1M/sift_query.fvecs")
d = xt.shape[1]

print("load GT")

gt = ivecs_read("../sift1M/sift_groundtruth.ivecs")
gt = gt.astype('int64') # note: cast in numpy has different meaning from cast in C!
k = gt.shape[1]

print("prepare criterion")
crit = faiss.IntersectionCriterion(xq.shape[0], k) # recall@K criterion
crit.set_groundtruth(None, gt) # set gt for oracle early termination
crit.nnn = k

# Retrieve HNSW stats
index_key = "IVF4096,Flat"
# index_key = "HNSW64,PQ32"
index = faiss.index_factory(d, index_key)

index.train(xt)
index.add(xb)

print("finished first add")

# index.hnsw.prune_state = 0
# index.add(xb)

# print("finished second pruned add\n")

# k=gt.shape[1]

# t0 = time.time()

# for ef in []:
#     index.hnsw.efSearch = ef
#     faiss.cvar.hnsw_stats.reset()
#     index.nprobe = 16

#     D, I = index.search(xq, k) # sanity check
#     recall = crit.evaluate(D, I)
#     latency = time.time() - t0

#     print(f"{latency:.3f}, {recall:.3f}")


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
    

# size_in_bytes = get_hnsw_pq_memory_breakdown(index)
# print(f"Memory Usage: {size_in_bytes / (1024**2):.2f} MB")

chunk = faiss.serialize_index(index)
print(f"Index size: {len(chunk) / (1024**2):.2f} MB")

