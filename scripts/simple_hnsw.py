import numpy as np
import faiss

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

# Retrieve HNSW stats
index_key = "HNSW64"
index = faiss.index_factory(d, index_key)

index.train(xt)
index.add(xb)

k=gt.shape[1]

D, I = index.search(xq, k) # sanity check
print(I)
print(D)