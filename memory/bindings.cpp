#include <pybind11/pybind11.h>
// #include <cuda_runtime_api.h>

namespace py = pybind11;

extern "C" {
    void init_allocator(size_t pool_size_bytes, size_t block_size_bytes);
    void cuda_free(void* ptr);
    void* cuda_malloc(ssize_t size);
    void free_all();
    void dump_free();
    void dump_allocs();
    void dump_consumption();
    void destroy_allocator();
} // extern "C"


PYBIND11_MODULE(tqpmemory, m) {
    m.def("init", &init_allocator, "Init GPU pooled allocator");
    m.def("destroy", &destroy_allocator, "Init GPU pooled allocator");
    m.def("malloc", &cuda_malloc, "Memory alloc");
    m.def("free", &cuda_free, "Free");
    m.def("free_all", &free_all, "Free all alloc'd mem");
    m.def("dump_free", &dump_free, "Dump free list");
    m.def("dump_allocs", &dump_allocs, "Dump alloc'd list");
    m.def("dump_consumption", &dump_consumption, "Dump mem usage");
}