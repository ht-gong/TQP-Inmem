# ------------------------------------------------------------
# CrossTorchGPU.cmake
# - Finds Torch from either libtorch or a Python wheel
# - Detects HIP/ROCm and/or CUDA
# - Chooses a backend automatically (HIP, CUDA, or CPU)
# - Exposes cross_torch_gpu_add_example() that builds an example
#
# Cache knobs (optional):
#   TORCH_HINT         : Path to torch root (libtorch or site-packages/torch)
#   FORCE_BACKEND      : AUTO (default), CUDA, HIP, CPU
#   HIP_ROOT_DIR       : ROCm root (e.g., /opt/rocm)
#
# Exposed variables:
#   TORCH_FOUND, TORCH_ROOT, TORCH_VERSION
#   HIP_FOUND, HIP_ROOT_DIR
#   CUDA_FOUND, CUDAToolkit_VERSION
#   CROSS_BACKEND      : HIP/CUDA/CPU
#
# ------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
include_guard(GLOBAL)

# --- 1. Find PyTorch ---
# We prioritize TORCH_HINT, then try to find Torch via its Python package,
# and finally let CMake search default paths.
if(NOT Torch_FOUND)
  set(_torch_search_paths)
  if(DEFINED TORCH_HINT)
    list(APPEND _torch_search_paths "${TORCH_HINT}")
  endif()

  # A robust way to find torch from a Python install is to ask torch itself.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE _torch_cmake_prefix
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch, sys; lib=(pathlib.Path(torch.__file__).resolve().parent/'lib'); print(lib)"
      OUTPUT_VARIABLE TORCH_PY_LIB_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_library(TORCH_PYTHON_LIBRARY
      NAMES torch_python
      PATHS "${TORCH_PY_LIB_DIR}" "${TORCH_INSTALL_PREFIX}/lib"
    )
    if(NOT TORCH_PYTHON_LIBRARY)
      message(FATAL_ERROR "Set -DTORCH_PYTHON_LIBRARY=/full/path/to/libtorch_python.so")
    endif()
    if(IS_DIRECTORY "${_torch_cmake_prefix}")
      list(APPEND _torch_search_paths "${_torch_cmake_prefix}")
    endif()
  endif()

  find_package(Torch HINTS ${_torch_search_paths})
endif()

if(NOT Torch_FOUND)
  message(FATAL_ERROR "Torch not found. Set -DTORCH_HINT to your libtorch root or Python's site-packages/torch directory.")
endif()

# Expose TORCH_ROOT derived from the found Torch_DIR
get_filename_component(TORCH_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
message(STATUS "CrossTorchGPU: Torch found: ${Torch_FOUND} | Version: ${Torch_VERSION} | Root: ${TORCH_ROOT}")


# --- 2. Detect Available GPU Backends on the System ---
# Find HIP/ROCm
find_package(HIP QUIET)
if(HIP_FOUND)
  message(STATUS "CrossTorchGPU: HIP/ROCm found: ${HIP_FOUND} | Root: ${HIP_ROOT_DIR}")
else()
  message(STATUS "CrossTorchGPU: HIP/ROCm not found.")
endif()

# Find CUDA
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(CUDA_FOUND TRUE) # find_package(CUDAToolkit) doesn't set CUDA_FOUND
  message(STATUS "CrossTorchGPU: CUDA Toolkit found: ${CUDA_FOUND} | Version: ${CUDAToolkit_VERSION}")
else()
  set(CUDA_FOUND FALSE)
  message(STATUS "CrossTorchGPU: CUDA Toolkit not found.")
endif()


# --- 3. Determine PyTorch Binary Flavor ---
# Check if the found PyTorch was built with CUDA or HIP support.
set(_torch_bin_flavor "CPU")
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print('CUDA' if torch.version.cuda else ('HIP' if getattr(torch.version, 'hip', None) else 'CPU'))"
    OUTPUT_VARIABLE _torch_bin_flavor
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
message(STATUS "CrossTorchGPU: Detected Torch binary flavor: ${_torch_bin_flavor}")


# --- 4. Choose and Configure the Backend ---
set(FORCE_BACKEND "AUTO" CACHE STRING "Force a specific backend: AUTO, CUDA, HIP, CPU")
set_property(CACHE FORCE_BACKEND PROPERTY STRINGS AUTO CUDA HIP CPU)

set(chosen_backend "CPU")
if(FORCE_BACKEND STREQUAL "CUDA")
  if(CUDA_FOUND)
    set(chosen_backend "CUDA")
  else()
    message(WARNING "FORCE_BACKEND=CUDA, but CUDA Toolkit was not found. Falling back to CPU.")
  endif()
elseif(FORCE_BACKEND STREQUAL "HIP")
  if(HIP_FOUND)
    set(chosen_backend "HIP")
  else()
    message(WARNING "FORCE_BACKEND=HIP, but HIP/ROCm was not found. Falling back to CPU.")
  endif()
elseif(FORCE_BACKEND STREQUAL "CPU")
  set(chosen_backend "CPU")
else() # AUTO logic
  if(_torch_bin_flavor STREQUAL "HIP" AND HIP_FOUND)
    set(chosen_backend "HIP")
  elseif(_torch_bin_flavor STREQUAL "CUDA" AND CUDA_FOUND)
    set(chosen_backend "CUDA")
  elseif(HIP_FOUND) # Fallback if flavor detection fails but hardware exists
    message(STATUS "CrossTorchGPU: Torch flavor is not HIP, but found HIP hardware. Attempting HIP backend.")
    set(chosen_backend "HIP")
  elseif(CUDA_FOUND) # Fallback if flavor detection fails but hardware exists
    message(STATUS "CrossTorchGPU: Torch flavor is not CUDA, but found CUDA hardware. Attempting CUDA backend.")
    set(chosen_backend "CUDA")
  endif()
endif()

# Enable the language for the chosen backend
if(chosen_backend STREQUAL "HIP")
  # For modern ROCm, we must prefer amdclang++/clang++ over the legacy hipcc wrapper.
  if(NOT CMAKE_HIP_COMPILER)
    find_program(CMAKE_HIP_COMPILER NAMES amdclang++ clang++ HINTS "${HIP_ROOT_DIR}/bin" "${HIP_ROOT_DIR}/llvm/bin")
  endif()
  enable_language(HIP)
  set(CROSS_BACKEND "HIP")
elseif(chosen_backend STREQUAL "CUDA")
  enable_language(CUDA)
  set(CROSS_BACKEND "CUDA")
else()
  set(CROSS_BACKEND "CPU")
endif()

message(STATUS "CrossTorchGPU: Selected backend = ${CROSS_BACKEND}")


# --- 5. Public API Function ---
# Adds an example executable and links it against Torch.
function(cross_torch_gpu_add_example)
  set(options)
  set(oneValueArgs TARGET SOURCE)
  cmake_parse_arguments(X "${options}" "${oneValueArgs}" "" ${ARGN})

  if(NOT X_TARGET)
    set(X_TARGET example)
  endif()
  if(NOT X_SOURCE)
    set(X_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/example.cpp")
  endif()
  if(NOT EXISTS "${X_SOURCE}")
    message(FATAL_ERROR "Example source file not found at: ${X_SOURCE}")
  endif()

  # Set modern C++ standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  # PyTorch libraries are built with PIC, so executables linking them should be too.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  add_executable(${X_TARGET} "${X_SOURCE}")

  # The key step: Link against Torch.
  # We must be defensive here, as not all Torch versions export the modern
  # `Torch::torch` imported target. We check for the target first, and if
  # it doesn't exist, we fall back to the older `TORCH_LIBRARIES` variable.
  if(TARGET Torch::torch)
    message(STATUS "Linking against imported target: Torch::torch")
    # Modern way: The imported target handles includes, libraries, and dependencies.
    target_link_libraries(${X_TARGET} PRIVATE Torch::torch)
  elseif(DEFINED TORCH_LIBRARIES)
    message(STATUS "Linking against legacy variables: TORCH_LIBRARIES")
    # Old way: Manually add include directories and link libraries.
    target_include_directories(${X_TARGET} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${X_TARGET} PRIVATE ${TORCH_LIBRARIES})
  else()
    message(FATAL_ERROR "Could not link against Torch. After finding the package, neither the "
                        "imported target 'Torch::torch' nor the variable 'TORCH_LIBRARIES' was found. "
                        "Please check your Torch installation's CMake support files.")
  endif()

  # The torch_python library is for C++ extensions and may not always exist
  # as a separate target. Link it if available.
  if(TARGET Torch::torch_python)
    target_link_libraries(${X_TARGET} PRIVATE Torch::torch_python)
  endif()

  # Add a compile definition to control backend-specific code in the example.
  if(CROSS_BACKEND STREQUAL "HIP")
    target_compile_definitions(${X_TARGET} PRIVATE USE_HIP_BACKEND=1)
  elseif(CROSS_BACKEND STREQUAL "CUDA")
    target_compile_definitions(${X_TARGET} PRIVATE USE_CUDA_BACKEND=1)
  else()
    target_compile_definitions(${X_TARGET} PRIVATE USE_CPU_BACKEND=1)
  endif()

  # Set output directories for cleaner build trees
  set_target_properties(${X_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
endfunction()