import numpy as np

# for npy files
emb = np.load("wiki_subset.npy")
print(f"Original Data Type: {emb.dtype}")

print(emb)
print(emb.shape)

print(emb[0])

# for sift1M
# def ivecs_read(fname):
#     a = np.fromfile(fname, dtype="int32")
#     d = a[0]
#     return a.reshape(-1, d + 1)[:, 1:].copy()

# def fvecs_read(fname):
#     return ivecs_read(fname).view('float32')

# print("load data")

# xb = fvecs_read("../sift1M/sift_base.fvecs")
# print(xb[10])