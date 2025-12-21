#include <iostream>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>

struct LeannMappedEmbeddings {
    int fd;
    size_t file_size;
    const uint16_t* raw_ptr; // Points to raw 16-bit data
    size_t dim;      // Dimensions per vector

    mutable std::vector<float> conversion_buffer;

    LeannMappedEmbeddings(const char* filename, size_t d) : dim(d), conversion_buffer(d) {
        fd = open(filename, O_RDONLY);
        if (fd == -1) {
            perror("Error opening embedding file");
            exit(1);
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            perror("Error getting file size");
            exit(1);
        }
        file_size = sb.st_size;

        // Map the file as read-only. 
        // MAP_SHARED allows other processes to see it (standard for read-only).
        void* map = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            perror("Error mmapping file");
            exit(1);
        }

        raw_ptr = static_cast<const uint16_t*>(map);
        
        // Performance Hint: Tell OS we will access this randomly
        madvise(map, file_size, MADV_RANDOM);
    }

    ~LeannMappedEmbeddings() {
        if (raw_ptr) munmap((void*)raw_ptr, file_size);
        if (fd != -1) close(fd);
    }

    // Helper: Upcast single FP16 to FP32
    static float half_to_float(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x0001;
        uint32_t exp  = (h >> 10) & 0x001f;
        uint32_t mant = h & 0x03ff;
        uint32_t f_bits;

        if (exp == 0) {
            f_bits = (mant == 0) ? (sign << 31) : (sign << 31) | (mant << 13); // Denormal/Zero
        } else if (exp == 31) {
            f_bits = (sign << 31) | 0x7f800000 | (mant << 13); // Inf/NaN
        } else {
            f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        float result;
        std::memcpy(&result, &f_bits, sizeof(result));
        return result;
    }

    const float* get(size_t i) const {
        const uint16_t* src = raw_ptr + (i * dim);
        
        // Convert the specific vector on demand
        for (size_t j = 0; j < dim; j++) {
            conversion_buffer[j] = half_to_float(src[j]);
        }
        
        return conversion_buffer.data();
    }
};

int main() {
    LeannMappedEmbeddings embed_store("/mnt/local/yongye/faiss/scripts/wiki_subset.bin", 768);
    // LeannMappedEmbeddings embed_store("/mnt/local/yongye/faiss/sift1M/sift_base_fp16.bin", 128);

    size_t index = 0;  // change this to pick a different vector

    const float* emb = embed_store.get(index);

    std::cout << "Embedding " << index << ": ";
    for (size_t i = 0; i < embed_store.dim; i++) {
        std::cout << emb[i] << " ";
    }
    std::cout << "\n";

    return 0;
}