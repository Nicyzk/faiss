import pickle

import numpy as np

file="/mnt/local/yongye/faiss/data/embeddings/rpj_wiki/passages_00.pkl"

def read_and_reduce():
    with open(file, "rb") as f:
        emb = pickle.load(f)
    np.save('embeddings.npy', emb[1][:200000])


def main():
    read_and_reduce()

if __name__ == "__main__":
    main()


