#include <torch/extension.h>
#include <torch/library.h>
#include <iostream>
#include <io-sched.h>
#include <pybind11/stl.h>

using namespace at;

#define HIP_CHECK_ERROR(call) { \
  hipError_t error = call; \
  if (error != hipSuccess) { \
      std::cerr << "HIP Error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl; \
      exit(error); \
  } \
}


struct ExchangeWrapper {
  gpuio::sched::dyn::LoadBalancedExchange exchange;
  // gpuio::hip::MemoryRef dstDeviceRef;
  // py::list dstDeviceList;
  // py::list dstHostList;
  // gpuio::hip::MemoryRef dstHostRef;

  ExchangeWrapper(size_t size) : exchange(size) {}  

  std::vector<gpuio::hip::MemoryRef> tensorsToVector(py::list lst) {
    std::vector<gpuio::hip::MemoryRef> vec;
    // if(py::len(lst) == 1) {
    //   torch::Tensor t = lst[0].cast<torch::Tensor>();
    //   return gpuio::hip::MemoryRef{(uint8_t *) t.data_ptr(),
    //      (size_t) t.numel() * t.element_size(), t.device().index()};
    // }
    // size_t total_size = 0;
    for(py::handle i : lst) {
      torch::Tensor it = i.cast<torch::Tensor>();
      if (!it.is_contiguous()) {
        std::cout << "Warning: Tensor is not contiguous!" << std::endl;
      }
      // std::cout << "it.numel() * it.element_size() = " << it.numel() << " * " << it.element_size() << std::endl;
      vec.push_back(gpuio::hip::MemoryRef{
        reinterpret_cast<uint8_t*>(it.data_ptr()),
        static_cast<size_t>(it.numel() * it.element_size()),
        it.device().index()
      });
    }
    return vec;
    // uint8_t* hostDataPtr = nullptr;
    // HIP_CHECK_ERROR(hipHostMalloc(&hostDataPtr, total_size, hipHostMallocDefault)); 
    // uint8_t* curPtr = hostDataPtr;
    // for(py::handle i : lst) {
    //   torch::Tensor it = i.cast<torch::Tensor>();
    //   size_t data_size = (size_t) it.numel() * it.element_size(); 
    //   HIP_CHECK_ERROR(hipMemcpy(curPtr, it.data_ptr(), data_size, hipMemcpyDefault));
    //   curPtr = curPtr + data_size;
    // }
    // return gpuio::hip::MemoryRef{curPtr, total_size, lst[0].cast<torch::Tensor>().device().index()};
  }

  // void recoverTensors(std::vector<gpuio::hip::MemoryRef> mem_refs, py::list lst) { 
  //   uint8_t* curPtr = mem_ref.ptr;
  //   for(py::handle i : lst) {
  //     torch::Tensor it = i.cast<torch::Tensor>();
  //     size_t d_sz = (size_t) it.numel() * it.element_size();
  //     HIP_CHECK_ERROR(hipMemcpy(it.data_ptr(), curPtr, d_sz, hipMemcpyDefault));
  //     curPtr += d_sz;
  //   }
  //   if(py::len(lst) > 1) {
  //     free(mem_ref);
  //   // }
  //   for(int i = 0; i < mem_refs.size(); i++) {
  //     torch::Tensor it = lst[i].cast<torch::Tensor>();
  //   }
  // }

  void launch(py::list dstDevice, py::list srcHost, py::list dstHost, py::list srcDevice) {

    auto dstDevice_ = tensorsToVector(dstDevice);
    auto srcHost_ = tensorsToVector(srcHost);
    auto srcDevice_ = tensorsToVector(srcDevice);
    auto dstHost_ = tensorsToVector(dstHost);

    // this->dstDeviceRef = dstDevice_;
    // this->dstHostRef = dstHost_;

    exchange.launch(dstDevice_, srcHost_, dstHost_, srcDevice_);
  }

  // void launch(torch::Tensor dstDevice, torch::Tensor srcHost, torch::Tensor dstHost, torch::Tensor srcDevice) {
  //   auto dstDevice_ = gpuio::hip::MemoryRef{(uint8_t *) dstDevice.data_ptr(), (size_t) dstDevice.numel() * dstDevice.element_size(), dstDevice.device().index()};
  //   auto srcHost_ = gpuio::hip::MemoryRef{(uint8_t *) srcHost.data_ptr(), (size_t) srcHost.numel() * srcHost.element_size(), -1};
  //   auto dstHost_ = gpuio::hip::MemoryRef{(uint8_t *) dstHost.data_ptr(), (size_t) dstHost.numel() * dstHost.element_size(), -1};
  //   auto srcDevice_ = gpuio::hip::MemoryRef{(uint8_t *) srcDevice.data_ptr(), (size_t) srcDevice.numel() * srcDevice.element_size(), srcDevice.device().index()};
  //   exchange.launch(dstDevice_, srcHost_, dstHost_, srcDevice_);
  //   is_list = false;
  // }

  void sync() {
    exchange.sync();
  }
};


PYBIND11_MODULE(vortex, m) {
  py::class_<ExchangeWrapper>(m, "Exchange")
  .def(py::init<size_t>())
  .def("launch", &ExchangeWrapper::launch)
  .def("sync", &ExchangeWrapper::sync);
}
