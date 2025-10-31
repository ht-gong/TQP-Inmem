# --- Define the Python Module ---
if(CROSS_BACKEND STREQUAL "HIP")
  Python3_add_library(zero_copy_reader MODULE
    zero_copy_reader.hip
    WITH_SOABI)
elseif(CROSS_BACKEND STREQUAL "CUDA")
  Python3_add_library(zero_copy_reader MODULE
    zero_copy_reader.cu
    WITH_SOABI)
else()
   message(FATAL_ERROR "Cannot build without HIP or CUDA")
endif()

target_link_libraries(zero_copy_reader PRIVATE Python3::Module pybind11::module)
# Link against Torch. Prefer imported targets if available.
if(TARGET Torch::Torch)
  target_link_libraries(zero_copy_reader PRIVATE Torch::Torch)
elseif(TARGET torch)
  target_link_libraries(zero_copy_reader PRIVATE torch)
elseif(DEFINED TORCH_LIBRARIES)
  target_link_libraries(zero_copy_reader PRIVATE ${TORCH_LIBRARIES})
else()
  message(FATAL_ERROR "Torch not available while building zero_copy_reader")
endif()

if (TARGET Torch::Python)
  target_link_libraries(zero_copy_reader PRIVATE Torch::Python)
elseif(DEFINED TORCH_PYTHON_LIBRARY)
  target_link_libraries(zero_copy_reader PRIVATE ${TORCH_PYTHON_LIBRARY})
else()
  message(FATAL_ERROR "Torch Python lib not available while building zero_copy_reader")
endif()

# target_include_directories(zero_copy_reader PRIVATE $ENV{CONDA_PREFIX}/include) 
# NEEDED on conda for some reason


if(CROSS_BACKEND STREQUAL "HIP")
    target_compile_definitions(zero_copy_reader PRIVATE USE_HIP_BACKEND=1)
elseif(CROSS_BACKEND STREQUAL "CUDA")
    target_compile_definitions(zero_copy_reader PRIVATE USE_CUDA_BACKEND=1)
else()
    target_compile_definitions(zero_copy_reader PRIVATE USE_CPU_BACKEND=1)
endif()

set_target_properties(zero_copy_reader PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)



# # --- Configure Target with Discovered Dependencies ---

# # Make sure we can include torch headers (ATen/pybind shims)
# target_include_directories(cuda_zero_copy_reader PRIVATE ${TORCH_INCLUDE_DIRS})

# # Append Torch's recommended compile flags to CMake's global flags
# if(DEFINED TORCH_CXX_FLAGS)
#   string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
#   string(APPEND CMAKE_CUDA_FLAGS " ${TORCH_CXX_FLAGS}")
# endif()

# # --- robust way to link libtorch_python ---
# set(_torch_python_lib "")
# if (TARGET Torch::Python)
#   set(_torch_python_lib Torch::Python)
# else()
#   # Ask Python where torch lives, then look for libtorch_python in its lib dir
#   execute_process(
#     COMMAND ${Python3_EXECUTABLE} -c "import pathlib, torch, sys; lib=(pathlib.Path(torch.__file__).resolve().parent/'lib'); print(lib)"
#     OUTPUT_VARIABLE TORCH_PY_LIB_DIR
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#   )
#   find_library(TORCH_PYTHON_LIBRARY
#     NAMES torch_python
#     PATHS "${TORCH_PY_LIB_DIR}" "${TORCH_INSTALL_PREFIX}/lib"
#   )
#   if(NOT TORCH_PYTHON_LIBRARY)
#     message(FATAL_ERROR "Set -DTORCH_PYTHON_LIBRARY=/full/path/to/libtorch_python.so")
#   endif()
#   set(_torch_python_lib "${TORCH_PYTHON_LIBRARY}")

#   # Ensure runtime can find it (RPATH)
#   set_target_properties(cuda_zero_copy_reader PROPERTIES
#     BUILD_RPATH "${TORCH_PY_LIB_DIR}"
#     INSTALL_RPATH "${TORCH_PY_LIB_DIR}"
#   )
# endif()

# message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
# message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
# message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")


# target_link_libraries(cuda_zero_copy_reader PRIVATE Python3::Module pybind11::module ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# # --- Set Target Properties ---

# set_target_properties(cuda_zero_copy_reader PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
#   POSITION_INDEPENDENT_CODE ON
#   CXX_VISIBILITY_PRESET hidden
#   VISIBILITY_INLINES_HIDDEN ON
#   PREFIX ""  # produce cuda_zero_copy_reader.*.so (not lib*)
# )

# # --- Configure Output Location (drop .so in build/) ---

# set_target_properties(cuda_zero_copy_reader PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )

# foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
#   string(TOUPPER "${cfg}" CFGU)
#   set_target_properties(cuda_zero_copy_reader PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY_${CFGU} "${CMAKE_BINARY_DIR}"
#     RUNTIME_OUTPUT_DIRECTORY_${CFGU} "${CMAKE_BINARY_DIR}"
#   )
# endforeach()

# # Help the loader find LibTorch at runtime (alternative to setting LD_LIBRARY_PATH)
# if(DEFINED TORCH_INSTALL_PREFIX)
#   set(_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib")
#   set_target_properties(cuda_zero_copy_reader PROPERTIES
#     BUILD_RPATH "${_torch_lib_dir}"
#     INSTALL_RPATH "${_torch_lib_dir}"
#   )
# endif()
