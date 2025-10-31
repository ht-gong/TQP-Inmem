#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <torch/extension.h> 
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) do {                                 \
  cudaError_t _e = (x);                                     \
  if (_e != cudaSuccess) {                                  \
    fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",          \
            __FILE__, __LINE__, cudaGetErrorString(_e), _e);\
    std::abort();                                           \
  }                                                         \
} while (0)

#define CONST_NULL -64

namespace {

    template <typename T>
    __global__ void zero_copy_mask_host_alias(const T* __restrict__ h_alias,
                                     const int64_t* __restrict__ indices,
                                     size_t other_dims,
                                     size_t n,
                                     T* __restrict__ result)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = gridDim.x * blockDim.x;
        for (size_t i = tid; i < n; i += total) { 
            size_t index = i / other_dims;
            size_t offset = i % other_dims;
            result[i] = h_alias[indices[index] * other_dims + offset];
        }
    }

    template <typename T>
    inline void launch_copy_kernel(torch::Tensor host, torch::Tensor mask, torch::Tensor result, size_t other_dims) {
        auto indices = torch::nonzero(mask).squeeze(1);
        TORCH_CHECK(indices.numel() == result.numel() / other_dims,
        "Expected mask tensor and result tensor to match, but got ",
        indices.numel(), " vs ", result.numel() / other_dims)
        const int64_t N = indices.numel() * other_dims;
        const int threads = 1024;
        const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        const int blocks = std::min<int>(int((N + threads - 1) / threads), sms * 32);
        auto stream = at::cuda::getCurrentCUDAStream();
        
        T* h = host.data_ptr<T>();
        T* h_alias = nullptr;  // device-visible alias to h_data
        CHECK_CUDA(cudaHostGetDevicePointer((void**)&h_alias, (void*)h, 0));
        zero_copy_mask_host_alias<T><<<blocks, threads, 0, stream>>>(h_alias, indices.data_ptr<int64_t>(), other_dims, N, result.data_ptr<T>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    
    void zero_copy_scan_with_mask(torch::Tensor host,
                                   torch::Tensor mask,
                                   torch::Tensor result)
    {
        cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        if (!prop.canMapHostMemory) {
            std::fprintf(stderr,"This GPU cannot map host memory (no zero-copy support).\n");
            throw std::runtime_error("No mapping capabilities");
        }
        if (host.numel() == 0)
            return;
        TORCH_CHECK(host.is_cpu(), "expected host on CPU");
        TORCH_CHECK(host.is_pinned(), "expected host pinned");
        TORCH_CHECK(mask.is_cuda(), "expected mask on GPU");
        TORCH_CHECK(result.is_cuda(), "expected result on GPU");
        TORCH_CHECK(host.is_contiguous(), "make contiguous() or use accessors/strides");
        TORCH_CHECK(mask.is_contiguous(), "make contiguous() or use accessors/strides");
        TORCH_CHECK(result.is_contiguous(), "make contiguous() or use accessors/strides");

        std::vector<int64_t> dims = host.sizes().vec();
        int64_t other_dims = std::max(std::accumulate(dims.begin()+1, dims.end(), 0), 1); // Default is 1 in the case where there's only 1 dimension

        TORCH_CHECK(host.numel() % other_dims == 0, "host numel/dim mismatch");
        // TORCH_CHECK(mask.numel() == result.numel() / other_dims,
        //     "Expected mask tensor and result tensor to match, but got ",
        //     mask.numel(), " vs ", result.numel() / other_dims);
        TORCH_CHECK(mask.scalar_type() == at::kBool);
        TORCH_CHECK(mask.dim() == 1, "mask must be 1-D");


        switch (host.scalar_type()) {
            case at::kByte:   launch_copy_kernel<uint8_t >(host, mask, result, other_dims); break;
            case at::kChar:   launch_copy_kernel<int8_t  >(host, mask, result, other_dims); break;
            case at::kShort:  launch_copy_kernel<int16_t >(host, mask, result, other_dims); break;
            case at::kInt:    launch_copy_kernel<int32_t >(host, mask, result, other_dims); break;
            case at::kLong:   launch_copy_kernel<int64_t >(host, mask, result, other_dims); break;
            case at::kFloat:  launch_copy_kernel<float   >(host, mask, result, other_dims); break;
            case at::kDouble: launch_copy_kernel<double  >(host, mask, result, other_dims); break;
            default:
              TORCH_CHECK(false, "Unsupported dtype for this copy (bool/half/bfloat16/complex not enabled)");
        }
    }

    template <typename T>
    __global__ void zero_copy_rearrange_host_alias(const T* __restrict__ h_alias,
                                                 const int64_t* __restrict__ indices,
                                                 const int64_t* __restrict__ dests,
                                                 size_t other_dims,
                                                 size_t N,
                                                 T* __restrict__ result)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = gridDim.x * blockDim.x;
        for (size_t i = tid; i < N; i += total) { 
            size_t index = i / other_dims;
            size_t offset = i % other_dims;
            if(dests[index] == CONST_NULL) {
                result[dests[index] * other_dims + offset] = CONST_NULL;
                continue;
            }
            result[dests[index] * other_dims + offset] = h_alias[indices[index] * other_dims + offset];
        }
    }

    template <typename T>
    inline void launch_rearrange_kernel(torch::Tensor host, torch::Tensor rearrange, 
                                        torch::Tensor result, size_t other_dims) {
        const size_t N = rearrange.numel() * other_dims;
        // const size_t rows = host.numel() / other_dims;
        const int threads = 1024;
        const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        const int blocks = std::min<int>(int((N + threads - 1) / threads), sms * 32);
        auto stream = at::cuda::getCurrentCUDAStream();


        T* h = host.data_ptr<T>();
        T* h_alias = nullptr;  // device-visible alias to h_data
        CHECK_CUDA(cudaHostGetDevicePointer((void**)&h_alias, (void*)h, 0));
        auto sorted_rearrange = torch::sort(rearrange, true, -1, false);
        auto indices = std::get<0>(sorted_rearrange);
        auto perm = std::get<1>(sorted_rearrange);
        zero_copy_rearrange_host_alias<T><<<blocks, threads, 0, stream>>>(h_alias, 
                                                                          indices.data_ptr<int64_t>(), 
                                                                          perm.data_ptr<int64_t>(),
                                                                          other_dims, 
                                                                          N,
                                                                          result.data_ptr<T>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

    }
    
    void zero_copy_scan_with_rearrange(torch::Tensor host,
                                       torch::Tensor rearrange,
                                       torch::Tensor result)
    {
        cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        if (!prop.canMapHostMemory) {
            std::fprintf(stderr,"This GPU cannot map host memory (no zero-copy support).\n");
            throw std::runtime_error("No mapping capabilities");
        }
        if (host.numel() == 0)
            return;
        TORCH_CHECK(host.is_cpu(), "expected host on CPU");
        TORCH_CHECK(host.is_pinned(), "expected host pinned");
        TORCH_CHECK(rearrange.is_cuda(), "expected mask on GPU");
        TORCH_CHECK(result.is_cuda(), "expected result on GPU");
        TORCH_CHECK(host.is_contiguous(), "make contiguous() or use accessors/strides");
        TORCH_CHECK(rearrange.is_contiguous(), "make contiguous() or use accessors/strides");
        TORCH_CHECK(result.is_contiguous(), "make contiguous() or use accessors/strides");
        TORCH_CHECK(rearrange.scalar_type() == at::kLong);

        std::vector<int64_t> dims = host.sizes().vec();
        int64_t other_dims = std::max(std::accumulate(dims.begin()+1, dims.end(), 0), 1); // Default is 1 in the case where there's only 1 dimension
        TORCH_CHECK(host.numel() % other_dims == 0, "host numel/dim mismatch");
        TORCH_CHECK(rearrange.numel() == result.numel() / other_dims,
            "Expected rearrange tensor and result tensor to match, but got ",
            rearrange.numel(), " vs ", result.numel() / other_dims);

        switch (host.scalar_type()) {
            case at::kByte:   launch_rearrange_kernel<uint8_t >(host, rearrange, result, other_dims); break;
            case at::kChar:   launch_rearrange_kernel<int8_t  >(host, rearrange, result, other_dims); break;
            case at::kShort:  launch_rearrange_kernel<int16_t >(host, rearrange, result, other_dims); break;
            case at::kInt:    launch_rearrange_kernel<int32_t >(host, rearrange, result, other_dims); break;
            case at::kLong:   launch_rearrange_kernel<int64_t >(host, rearrange, result, other_dims); break;
            case at::kFloat:  launch_rearrange_kernel<float   >(host, rearrange, result, other_dims); break;
            case at::kDouble: launch_rearrange_kernel<double  >(host, rearrange, result, other_dims); break;
            default:
              TORCH_CHECK(false, "Unsupported dtype for this copy (bool/half/bfloat16/complex not enabled)");
        }
    }

}

PYBIND11_MODULE(zero_copy_reader, m) {
    m.def("zero_copy_mask", &zero_copy_scan_with_mask, "zero copy with masking");
    m.def("zero_copy_rearrange", &zero_copy_scan_with_rearrange, "zero copy with masking");
}