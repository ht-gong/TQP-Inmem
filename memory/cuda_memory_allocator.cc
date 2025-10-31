// Compile time switch for backends
#if defined(USE_HIP_BACKEND)
  #include <hip/hip_runtime_api.h>   /* host-side HIP API */
#elif defined(USE_CUDA_BACKEND)
  #include <cuda_runtime_api.h>      /* host-side CUDA API */
#else
  #error "No backend defined. Make sure CMake sets one of USE_HIP_BACKEND / USE_CUDA_BACKEND / USE_CPU_BACKEND."
#endif

#include <map>
#include <mutex>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cstdarg>   // or <stdarg.h>
#include <cstdio>    // or <stdio.h>
#include <exception>
#include <cassert>

namespace {

// -------- Config --------
inline uint64_t read_env_u64(const char* name, uint64_t def) {
    if (const char* s = std::getenv(name)) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        if (end && *end == '\0') return static_cast<uint64_t>(v);
    }
    return def;
}

inline bool read_env_bool(const char* name) {
    if (const char* s = std::getenv(name)) {
        std::string v(s);
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return (v == "1" || v == "true" || v == "yes");
    }
    return false;
}

inline size_t byte_to_mb(size_t byte) {
    return byte >> 20;
}

static const uint64_t kSlabMB    = read_env_u64("PA_CUDA_SLAB_MB", 40960); // 20 GB default

class GPUMemoryPool {
public:
    GPUMemoryPool() : pool_size_bytes_(0), block_size_bytes_(0), mem_pool_(nullptr) {}

    ~GPUMemoryPool() {
        destroy_allocator();
    }

    template<typename Key, typename Value>
    void printMap(const std::map<Key, Value>& myMap) {
        std::cerr << "{ ";
        for (const auto& pair : myMap) {
            // You can customize the separator and format here
            std::cerr << pair.first << ": " << pair.second << ", ";
        }
        std::cerr << "}\n";
        std::cerr <<"DONTMINDME\n";
    }


    void init(size_t pool_size_bytes, size_t block_size_bytes) {

        block_size_bytes = (block_size_bytes >> 6) << 6;
        assert(block_size_bytes % 64 == 0); // Ensure 64 byte alignment always

        std::lock_guard<std::mutex> lock(mtx_);
        if (mem_pool_) return;

        pool_size_bytes_ = pool_size_bytes;
        block_size_bytes_ = block_size_bytes;
        if (pool_size_bytes_ < block_size_bytes_ || block_size_bytes_ == 0) {
            std::cerr << "Invalid pool/block size." << std::endl;
            return;
        }

        num_blocks_ = pool_size_bytes_ / block_size_bytes_;
        
        #if defined(USE_CUDA_BACKEND)
            cudaError_t status = cudaMalloc(&mem_pool_, pool_size_bytes_);
            if (status != cudaSuccess || !mem_pool_) {
                mem_pool_ = nullptr;
                std::cerr << "Failed to allocate pinned memory pool: " << cudaGetErrorString(status) << std::endl;
                return;
            }
        #else
            hipError_t status = hipMalloc(&mem_pool_, pool_size_bytes_);
            if (status != hipSuccess || !mem_pool_) {
                mem_pool_ = nullptr;
                std::cerr << "Failed to allocate device memory pool: "
                        << hipGetErrorString(status) << std::endl;
                return;
            }
        #endif

        free_blocks_.clear();
        free_blocks_[0] = num_blocks_;
        allocs_.clear();
    }

    void destroy_allocator() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (mem_pool_) {
            #if defined(USE_CUDA_BACKEND)
                cudaFree(mem_pool_);
            #else 
                hipFree(mem_pool_);
            #endif               
            mem_pool_ = nullptr;
        }
        pool_size_bytes_ = 0;
        block_size_bytes_ = 0;
        free_blocks_.clear();
        allocs_.clear();
    }

    void* malloc(size_t size) {
        if (!mem_pool_) return nullptr;
        if (size == 0) size = 1;
        size_t alloc_blocks = (size + block_size_bytes_ - 1) / block_size_bytes_;

        std::lock_guard<std::mutex> lock(mtx_);

        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= alloc_blocks) {
                size_t start_block = it->first;
                
                if (it->second == alloc_blocks) {
                    free_blocks_.erase(it);
                } else {
                    size_t len = it->second;
                    free_blocks_.erase(it);
                    free_blocks_[start_block + alloc_blocks] = len - alloc_blocks;
                }
                
                allocs_[start_block] = alloc_blocks;
                

                void* ptr = static_cast<char*>(mem_pool_) + start_block * block_size_bytes_;
                return ptr;
            }
        }
        
        std::cerr<<"Out Of Memory!"<<std::endl;
        throw std::runtime_error("Pool out of memory");
        return nullptr; // Out of memory
    }

    void free(void* ptr) {
        // std::cerr<<"Free"<<std::endl;
        if (!ptr || !mem_pool_) return;

        std::lock_guard<std::mutex> lock(mtx_);

        if (ptr < mem_pool_ || ptr >= static_cast<char*>(mem_pool_) + pool_size_bytes_) {
            return; // Pointer out of range
        }

        size_t offset = static_cast<char*>(ptr) - static_cast<char*>(mem_pool_);
        if (offset % block_size_bytes_ != 0) {
            return; // Invalid pointer
        }
        size_t start_block = offset / block_size_bytes_;

        auto it_alloc = allocs_.find(start_block);
        if (it_alloc == allocs_.end()) {
            return; // Not an allocated block
        }

        size_t num_blocks = it_alloc->second;
        allocs_.erase(it_alloc);

        // Coalesce
        size_t new_start = start_block;
        size_t new_len = num_blocks;

        auto next_it = free_blocks_.upper_bound(start_block);
        if (next_it != free_blocks_.end()) {
            if (start_block + num_blocks == next_it->first) {
                new_len += next_it->second;
                free_blocks_.erase(next_it);
            }
        }

        auto prev_it = free_blocks_.lower_bound(start_block);
        if (prev_it != free_blocks_.begin()) {
            --prev_it;
            if (prev_it->first + prev_it->second == start_block) {
                new_start = prev_it->first;
                new_len += prev_it->second;
                free_blocks_.erase(prev_it);
            }
        }

        free_blocks_[new_start] = new_len;
    }

    void free_all() {
        if (!mem_pool_) return;

        std::lock_guard<std::mutex> lock(mtx_);
        allocs_.clear();
        free_blocks_.clear();
        free_blocks_[0] = num_blocks_;
        
    }

    void dump_consumption() {
        size_t consumption = 0;
         for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            consumption += it->second * block_size_bytes_;
         }
         std::cerr << "Mem free: " << byte_to_mb(consumption)
                   << " out of " << byte_to_mb(pool_size_bytes_) << std::endl;
    }
    
    void dump_free() {
        printMap(free_blocks_);
    }

    void dump_allocs() {
        printMap(allocs_);
    }

private:
    size_t pool_size_bytes_;
    size_t block_size_bytes_;
    void* mem_pool_;
    size_t num_blocks_;

    std::map<size_t, size_t> free_blocks_; // start_block -> num_blocks
    std::map<size_t, size_t> allocs_; // start_block -> num_blocks

    std::mutex mtx_;
};

static GPUMemoryPool g_mem_pool;

} // namespace

extern "C" {

void init_allocator(size_t pool_size_bytes, size_t block_size_bytes) {
    g_mem_pool.init(pool_size_bytes, block_size_bytes);
}

#if defined(USE_HIP_BACKEND)
    void pa_cuda_free(void* ptr, ssize_t size, int device, hipStream_t /*stream*/) {
        g_mem_pool.free(ptr);
    }
#else 
    void pa_cuda_free(void* ptr, ssize_t size, int device, cudaStream_t /*stream*/) {
        g_mem_pool.free(ptr);
    }
#endif 

#if defined(USE_HIP_BACKEND)
    void* pa_cuda_malloc(ssize_t size, int device, hipStream_t /*stream*/) {
        return g_mem_pool.malloc(size);
    }
#else 
    void* pa_cuda_malloc(ssize_t size, int device, cudaStream_t /*stream*/) {
        return g_mem_pool.malloc(size);
    }
#endif

void cuda_free(void* ptr) {
    g_mem_pool.free(ptr);
}

void* cuda_malloc(ssize_t size) {
    return g_mem_pool.malloc(size);
}

void free_all() {
    g_mem_pool.free_all();
}

void dump_consumption() {
    g_mem_pool.dump_consumption();
}

void dump_free() {
    g_mem_pool.dump_free();
}

void dump_allocs() {
    g_mem_pool.dump_allocs();
}

void destroy_allocator() {
    g_mem_pool.destroy_allocator();
}

} // extern "C"

// PYBIND11_MODULE(cuda_mem_pool, cuda_mp) {
//     cuda_mp.def("init", &init_allocator, "Init GPU pooled allocator");
//     cuda_mp.def("destroy", &destroy_allocator, "Init GPU pooled allocator");
//     cuda_mp.def("free_all", &free_all, "Free all alloc'd mem");
//     cuda_mp.def("dump_free", &dump_free, "Dump free list");
//     cuda_mp.def("dump_allocs", &dump_allocs, "Dump alloc'd list");
// }
