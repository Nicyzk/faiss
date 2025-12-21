import numpy as np
import os

def convert_npy_to_bin(input_path, output_path):
    print(f"Loading {input_path}...")
    data = np.load(input_path)

    # 2. Print shape/dtype so you know what to write in C++
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")

    if data.dtype != "float16":
        print("Converting data to float16...")
        data = data.astype('float16')
        
    # 3. Write raw binary
    # This writes 2 bytes per element strictly
    data.tofile(output_path)
    print(f"Saved raw binary to {output_path}")

# convert from fp64 to fp16
def convert_npy_to_npy(input_path, output_path):
    # 2. Load the input file
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
    else:
        print(f"Loading {input_path}...")
        data = np.load(input_path)

        print(f"Original Shape: {data.shape}")
        print(f"Original Dtype: {data.dtype}")

        # 3. Convert to float16
        if data.dtype != 'float16':
            print("Converting to float16...")
            data_fp16 = data.astype('float16')
        else:
            print("Data is already float16.")
            data_fp16 = data

        # 4. Save as .npy (preserves shape and header info)
        print(f"Saving to {output_path}...")
        np.save(output_path, data_fp16)

        # 5. Verify
        print("Verification check:")
        check_data = np.load(output_path)
        print(f"New Dtype: {check_data.dtype}")
        print(f"New Shape: {check_data.shape}")
        print("Done!")

# only for converting SIFT1M from original fvecs to bin file so that C++ code can read
def convert_sift1m_to_bin(input_path, output_path):
    def fvecs_read(filename):
        a = np.fromfile(filename, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

    print("Reading fvecs...")
    data = fvecs_read(input_path)

    print("Converting to float16...")
    data_fp16 = data.astype('float16')

    print("Saving raw bin...")
    data_fp16.tofile(output_path)
    print("Done.")


convert_npy_to_bin("wiki_subset.npy", "wiki_subset.bin")
# convert_npy_to_npy("wiki_subset.npy", "subset_fp16.npy")
# convert_sift1m_to_bin("../sift1M/sift_base.fvecs", "../sift1M/sift_base_fp16.bin")


