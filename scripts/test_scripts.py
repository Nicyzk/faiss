import faiss
import numpy as np

def verify_custom_storage():
    # 1. Setup Parameters
    d = 64          # Dimension
    ntotal = 500    # Number of vectors
    M = 32          # HNSW neighbors
    pq_m = 8        # Product Quantization sub-quantizers
    
    print(f"Generating {ntotal} vectors of dimension {d}...")
    xt = np.random.rand(ntotal, d).astype('float32') # Training data
    xb = np.random.rand(ntotal, d).astype('float32') # Database data

    # 2. Create and Train IndexHNSWPQ
    print("Creating and training IndexHNSWPQ...")
    index = faiss.IndexHNSWPQ(d, pq_m, M)
    index.train(xt)

    # 3. Add Data (Triggers your modified C++ 'add' function)
    print("Adding data to index...")
    index.add(xb)

    # 4. Access Your Custom Field
    print("\nRetrieving custom storage from C++...")
    try:
        # [FIX] Convert the C++ std::vector to a numpy array
        # 'node_reconstructed_values' is the std::vector<float> exposed by SWIG
        raw_storage = faiss.vector_to_array(index.hnsw.node_reconstructed_values)
        
        # Reshape to (ntotal, d) for easy access
        # We use index.ntotal to ensure we match the actual index size
        custom_storage = raw_storage.reshape(index.ntotal, d)
        
    except AttributeError:
        print("\n[ERROR] 'node_reconstructed_values' not found in index.hnsw.")
        print("Did you recompile the Python bindings?")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to convert vector to array: {e}")
        return

    # 5. Verify Size
    expected_size = index.ntotal * d
    actual_size = raw_storage.size
    print(f"Verifying Storage Size:")
    print(f"  Expected elements: {expected_size}")
    print(f"  Actual elements:   {actual_size}")
    
    if expected_size != actual_size:
        print("  [FAIL] Size mismatch!")
        return

    # 6. Verify Content (Value Accuracy)
    print("\nVerifying Vector Values (checking first 3 vectors)...")
    is_correct = True
    
    for i in range(3):
        # A. Get value from your new custom storage (now a numpy array)
        stored_vec = custom_storage[i]
        
        # B. Get value from standard FAISS reconstruction (Gold Standard)
        reconstructed_vec = index.reconstruct(i)

        # C. Compare
        diff = np.linalg.norm(stored_vec - reconstructed_vec)
        print(f"  ID {i}: Difference = {diff:.6f}")
        
        if diff > 1e-5:
            print(f"    [FAIL] Vector {i} does not match!")
            print(f"    Stored: {stored_vec[:5]}...")
            print(f"    Recons: {reconstructed_vec[:5]}...")
            is_correct = False

    if is_correct:
        print("\n[SUCCESS] Your C++ modification is working correctly!")
    else:
        print("\n[FAIL] Data mismatch detected.")

if __name__ == "__main__":
    verify_custom_storage()