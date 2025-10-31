#include <rocprim/rocprim.hpp>
#include <torch/extension.h>
#include <torch/library.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

using namespace at;

template <typename K, typename V>
void mergehip(K* keys_input1, K* keys_input2, K* keys_output,
            V* values_input1, V* values_input2, V* values_output, size_t input_size1, size_t input_size2) {
    size_t need;
    rocprim::merge(
        nullptr, need,
        keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input_size1, input_size2
    );

    void *temp_ptr; 
    hipMalloc(&temp_ptr, need);
    
    auto err = rocprim::merge(temp_ptr, need,
        keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input_size1, input_size2
    );

    if (err != hipSuccess) {
        TORCH_CHECK("Unsuccessful Merge");
    }
    hipFree(temp_ptr);
}

template <typename K, typename V>
void dispatch_merge_second(K* k1, K* k2, K* kout,
    torch::Tensor values_input1, torch::Tensor values_input2, torch::Tensor values_output, 
    size_t input_size1, size_t input_size2) {
    V* v1 = values_input1.data_ptr<V>();
    V* v2 = values_input2.data_ptr<V>();
    V* vout = values_output.data_ptr<V>();
    mergehip<K, V>(k1, k2, kout, v1, v2, vout, input_size1, input_size2);
}

template <typename K>
void dispatch_merge_first(torch::Tensor keys_input1, torch::Tensor keys_input2, torch::Tensor keys_output,
    torch::Tensor values_input1, torch::Tensor values_input2, torch::Tensor values_output) {
    K* k1 = keys_input1.data_ptr<K>();
    K* k2 = keys_input2.data_ptr<K>();
    K* kout = keys_output.data_ptr<K>();
    if (values_input1.dtype() == torch::kFloat64) {
        dispatch_merge_second<K, double_t>(k1, k2, kout, values_input1, values_input2, values_output, keys_input1.sizes()[0], keys_input2.sizes()[0]);
    } else if (values_input1.dtype() == torch::kFloat32) {
        dispatch_merge_second<K, float_t>(k1, k2, kout, values_input1, values_input2, values_output, keys_input1.sizes()[0], keys_input2.sizes()[0]);
    } else if (values_input1.dtype() == torch::kInt64) {
        dispatch_merge_second<K, int64_t>(k1, k2, kout, values_input1, values_input2, values_output, keys_input1.sizes()[0], keys_input2.sizes()[0]);
    } else if (values_input1.dtype() == torch::kInt8) {
        dispatch_merge_second<K, int8_t>(k1, k2, kout, values_input1, values_input2, values_output, keys_input1.sizes()[0], keys_input2.sizes()[0]);
    } else {
        TORCH_CHECK(false, "Unsupported data type for merge");
    }
}

void merge(torch::Tensor keys_input1, torch::Tensor keys_input2, torch::Tensor keys_output,
    torch::Tensor values_input1, torch::Tensor values_input2, torch::Tensor values_output) {

    TORCH_CHECK(keys_input1.dtype() == keys_input2.dtype(), "Key tensors must have same dtype");
    TORCH_CHECK(values_input1.dtype() == values_input2.dtype(), "Value tensors must have same dtype");
    TORCH_CHECK(keys_input1.size(0) == values_input1.size(0), "Input key/value sizes mismatch");
    TORCH_CHECK(keys_input2.size(0) == values_input2.size(0), "Input key/value sizes mismatch");
    if (keys_input1.dtype() == torch::kFloat64) {
        dispatch_merge_first<double_t>(keys_input1, keys_input2, keys_output, values_input1, values_input2, values_output);
    } else if (keys_input1.dtype() == torch::kFloat32) {
        dispatch_merge_first<float_t>(keys_input1, keys_input2, keys_output, values_input1, values_input2, values_output);
    } else if (keys_input1.dtype() == torch::kInt64) {
        dispatch_merge_first<int64_t>(keys_input1, keys_input2, keys_output, values_input1, values_input2, values_output);
    } else if (keys_input1.dtype() == torch::kInt8) {
        dispatch_merge_first<int8_t>(keys_input1, keys_input2, keys_output, values_input1, values_input2, values_output);
    } else {
        TORCH_CHECK(false, "Unsupported data type for merge");
    }
}

PYBIND11_MODULE(hipTQP, m) {
    m.def("merge", &merge, "Merging 2 on device tensors");
}