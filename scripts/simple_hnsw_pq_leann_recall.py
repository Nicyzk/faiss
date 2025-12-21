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
# index_key = "HNSW64"
index_key = "HNSW64,PQ32"
# index_key = "HNSW24,PQ32" # Small M
# index_key = "IVF4096,PQ32"
index = faiss.index_factory(d, index_key)

index.train(xt)
index.add(xb)

print("finished first add")

index.hnsw.prune_state = 0
index.add(xb)

print("finished second pruned add\n")

k=gt.shape[1]

faiss.cvar.hnsw_stats.reset()
# index.nprobe = ef

t0 = time.time()
D, I = index.search(xq, k) # sanity check
recall = crit.evaluate(D, I)
latency = time.time() - t0

print(f"{latency:.3f}, {recall:.3f}")
