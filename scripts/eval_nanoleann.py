#!/usr/bin/env python3
"""Test only Faiss HNSW"""

import os
import sys
import time
import json
import psutil
import numpy as np

# choose in ["FLAT", "HNSW", "HNSWPQ"]. LEANN is 
index_type = "HNSWPQ"

def recall_at_k(gt_path: str, candidate_data, k: int):
    if not os.path.exists(gt_path):
        print(f"Ground truth file not found: {gt_path}")
        return {}

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    gt_map = {}
    for entry in gt_data:
        q_text = entry['query']
        # Extract content strings, keeping original order (which implies ranking)
        gt_map[q_text] = entry['results']

    total_recall = 0
    query_count = 0

    for cand_entry in candidate_data:
        query_text = cand_entry['query']

        if query_text not in gt_map:
            print(f"[WARN] Query not found in ground truth: '{query_text[:30]}...'")
            continue

        gt_contents_all = gt_map[query_text]
        cand_contents_all = cand_entry['results']
        
        query_count += 1

        # Calculate Recall for each K
        gt_set = set(gt_contents_all[:k])
        cand_set = set(cand_contents_all[:k])
        
        if not gt_set:
            recall = 0.0
        else:
            # Intersection count
            matches = len(gt_set.intersection(cand_set))
            recall = matches / len(gt_set)
        
        total_recall += recall

    # Compute Averages
    if query_count == 0:
        print("No matching queries found to evaluate.")
        return {}

    avg_recall = total_recall / query_count
    return avg_recall

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

class MemoryTracker:
    def __init__(self, name: str):
        self.name = name
        self.start_mem = get_memory_usage()
        self.stages = []

    def checkpoint(self, stage: str):
        current_mem = get_memory_usage()
        diff = current_mem - self.start_mem
        print(f"[{self.name} - {stage}] Memory: {current_mem:.1f} MB (+{diff:.1f} MB)")
        self.stages.append((stage, current_mem))
        return current_mem

    def summary(self):
        peak_mem = max(mem for _, mem in self.stages)
        print(f"Peak Memory: {peak_mem:.1f} MB")
        return peak_mem

def main():
    try:
        import faiss
    except ImportError:
        print("Faiss is not installed.")
        print(
            "Please install it with `uv pip install faiss-cpu` and you can  then run this script again"
        )
        sys.exit(1)

    from llama_index.core import (
        Settings,
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core.ingestion import IngestionPipeline

    tracker = MemoryTracker("Faiss HNSW")
    tracker.checkpoint("Initial")

    embed_model = HuggingFaceEmbedding(model_name="facebook/contriever")
    Settings.embed_model = embed_model
    tracker.checkpoint("After embedding model setup")

    train_embeddings = np.load('leann_vanilla_embeddings.npy', mmap_mode='r')

    d = 768
    if index_type == "FLAT":
        faiss_index = faiss.IndexFlatL2(d)
    
    if index_type == "HNSW":
        faiss_index = faiss.IndexHNSWFlat(d, 32)
        faiss_index.hnsw.efConstruction = 64

    if index_type == "HNSWPQ":
        faiss_index = faiss.IndexHNSWPQ(d, 32, 32)
        faiss_index.hnsw.efConstruction = 64
        faiss_index.train(train_embeddings)

    
    tracker.checkpoint("After Faiss index creation")

    documents = SimpleDirectoryReader(
        "doc_data",
        recursive=True,
        encoding="utf-8",
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()
    tracker.checkpoint("After document loading")

    # Parse into chunks using the same splitter as LEANN
    node_parser = SentenceSplitter(
        chunk_size=256, chunk_overlap=20, separator=" ", paragraph_separator="\n\n"
    )

    tracker.checkpoint("After text splitter setup")

    # Run the pipeline to get Node objects
    pipeline = IngestionPipeline(transformations=[node_parser])
    nodes = pipeline.run(documents)

    # Check if index already exists and try to load it
    index_loaded = False
    # if os.path.exists("./storage_faiss"):
    #     print("Loading existing Faiss HNSW index...")
    #     try:
    #         # Use the correct Faiss loading pattern from the example
    #         vector_store = FaissVectorStore.from_persist_dir("./storage_faiss")
    #         storage_context = StorageContext.from_defaults(
    #             vector_store=vector_store, persist_dir="./storage_faiss"
    #         )
    #         from llama_index.core import load_index_from_storage

    #         index = load_index_from_storage(storage_context=storage_context)
    #         print("Index loaded from ./storage_faiss")
    #         tracker.checkpoint("After loading existing index")
    #         index_loaded = True
    #     except Exception as e:
    #         print(f"Failed to load existing index: {e}")
    #         print("Cleaning up corrupted index and building new one...")
    #         # Clean up corrupted index
    #         import shutil

    #         if os.path.exists("./storage_faiss"):
    #             shutil.rmtree("./storage_faiss")

    if not index_loaded:
        # Clean up index in case we force regen
        import shutil

        if os.path.exists("./storage_faiss"):
            shutil.rmtree("./storage_faiss")

        print("Building new Faiss HNSW index...")

        # Use the correct Faiss building pattern from the example
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, transformations=[node_parser]
        )
        tracker.checkpoint("After index building")

        # Save index to disk using the correct pattern
        index.storage_context.persist(persist_dir="./storage_faiss")
        tracker.checkpoint("After index saving")

    # Measure runtime memory overhead
    print("\nMeasuring runtime memory overhead...")
    runtime_start_mem = get_memory_usage()
    print(f"Before load memory: {runtime_start_mem:.1f} MB")
    tracker.checkpoint("Before load memory")

    # WITH THIS:
    print("Initializing Retriever (skipping LLM synthesis)...")
    retriever = index.as_retriever(similarity_top_k=20)
    
    queries = [
        "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面,任务令一般在什么城市颁发",
        "What is LEANN and how does it work?",
        "华为诺亚方舟实验室的主要研究内容",
    ]

    json_output_data = []
    # query_embs = []

    for i, query in enumerate(queries):
        start_time = time.time()
        # query_emb = Settings.embed_model.get_query_embedding(query)
        # query_embs.append(query_emb)
        
        # .retrieve() runs the embedding + faiss search ONLY.
        # It returns a list of Node objects.
        nodes = retriever.retrieve(query)
        
        query_time = time.time() - start_time
        print(f"Query {i + 1} time: {query_time:.3f}s")
        tracker.checkpoint(f"After query {i + 1}")
        
        print(f"Query: {query}")

        query_record = {
            "query": query,
            "results": []
        }

        for i, node_with_score in enumerate(nodes):
            # 1. Access the similarity score
            score = node_with_score.score
            
            # 2. Access the actual underlying Node object
            real_node = node_with_score.node
            
            # 3. Get content and metadata for verification
            content = real_node.get_content()  # The text chunk
            metadata = real_node.metadata      # Dictionary with 'file_name', 'page_label', etc.
            node_id = real_node.node_id        # Unique ID of the chunk
            
            print("content:", content)
            query_record["results"].append(content)

        json_output_data.append(query_record)
        print("END OF ONE SEARCH")
    # ... rest of memory calculation ...
    # np.save("queries.npy", np.array(query_embs))

    runtime_end_mem = get_memory_usage()
    runtime_overhead = runtime_end_mem - runtime_start_mem

    peak_memory = tracker.summary()
    print(f"Peak Memory: {peak_memory:.1f} MB")
    print(f"Runtime Memory Overhead: {runtime_overhead:.1f} MB")

    # USE BELOW TO COMPUTE RECALL@K AGAINST A GT JSON FILE
    recall = recall_at_k("./eval_nanoleann_gt.json", json_output_data, 20)
    print(f"Recall@{20:<2}: {recall:.4f}")

    # USE BELOW TO PRINT GT TO OUTPUT JSON FILE
    try:
        json_output_file = "./eval_nanoleann_recompute_09.json"
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_output_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved search results to {json_output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()

