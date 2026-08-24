cmake_minimum_required(VERSION 3.16)
project(xstar_cpp LANGUAGES CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)              # 用 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # 强制,不许编译器降级
set(CMAKE_CXX_EXTENSIONS OFF)           # 不用 GNU 扩展,纯标准 C++
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # 生成 compile_commands.json,给 clangd 用(治编辑器 include 报错)

# CUDA 标准: 跟 C++ 对齐, 显式钉死不依赖 nvcc 默认(不同 toolkit 默认不同, 影响可复现)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "本地 sm_75(GTX 1660 Ti); 租卡 -DCMAKE_CUDA_ARCHITECTURES=75;80;90 覆盖")

# 找 Python 解释器(配置时用 -DPython_EXECUTABLE=... 指定 venv python)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# 找 CUDA Toolkit: 创建 CUDA::cudart 等 IMPORTED target
find_package(CUDAToolkit REQUIRED)

# 核心静态库:纯 C++ tensor/bfloat16(不碰 pybind11)
add_library(xstar_cpp_core STATIC
    xstar_cpp/src/tensor.cpp
    xstar_cpp/src/bfloat16.cpp
    xstar_cpp/src/mmap_file.cpp
    xstar_cpp/src/weight_io.cpp
    xstar_cpp/src/ops/rmsnorm.cpp
    xstar_cpp/src/ops/embedding.cpp
    xstar_cpp/src/ops/rope.cpp
    xstar_cpp/src/ops/matmul.cpp
    xstar_cpp/src/ops/linear.cpp
    xstar_cpp/src/ops/softmax.cpp
    xstar_cpp/src/ops/attention.cpp
    xstar_cpp/src/ops/gemm.cpp
    xstar_cpp/src/ops/mlp.cpp
    xstar_cpp/src/ops/transformer_block.cpp
    xstar_cpp/src/json_scan.cpp
    xstar_cpp/src/qwen2_config.cpp
    xstar_cpp/src/qwen2_model.cpp

    xstar_cpp/src/cuda/cuda_allocator.cu
    xstar_cpp/src/cuda/weight_load.cu
    
    xstar_cpp/src/ops/concat.cpp
    xstar_cpp/src/ops/concat.cu
    xstar_cpp/src/ops/rmsnorm.cu
    xstar_cpp/src/ops/softmax.cu
    xstar_cpp/src/ops/gemm.cu
    xstar_cpp/src/ops/attention.cu
    xstar_cpp/src/ops/mlp.cu
    xstar_cpp/src/ops/embedding.cu
    xstar_cpp/src/ops/rope.cu
    xstar_cpp/src/ops/head_split.cpp
    xstar_cpp/src/ops/head_split.cu
    xstar_cpp/src/ops/add.cpp
    xstar_cpp/src/ops/add.cu
    
    xstar_cpp/src/block_manager.cpp
    xstar_cpp/src/ops/attention_fa2.cpp
    xstar_cpp/src/ops/attention_fa2.cu
    xstar_cpp/src/kv_cache.cpp
    xstar_cpp/src/ops/paged_write.cpp
    xstar_cpp/src/ops/paged_write.cu
    xstar_cpp/src/paged_kv_cache.cpp
    xstar_cpp/src/ops/paged_attention.cpp
    xstar_cpp/src/ops/paged_attention.cu

    xstar_cpp/src/ops/paged_attention_decode.cu
    xstar_cpp/src/ops/paged_attention_prefill.cu
)
target_include_directories(xstar_cpp_core PUBLIC xstar_cpp/include)
# 链 CUDA runtime, PUBLIC 因静态库将来要透传给链它的 .so(用 PRIVATE 则 .so 链接阶段找不到 cudart)
target_link_libraries(xstar_cpp_core PUBLIC CUDA::cudart)
# 核心静态库开 PIC, 才能链进 .so(共享库)(否则链接报 "recompile with -fPIC")
set_target_properties(xstar_cpp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 定位 pybind11: 用上一步找到的 Python
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${PYBIND11_CMAKEDIR} REQUIRED)

# pybind11 模块: 绑定 .cpp 编成可 import 的 .so, 链核心库
pybind11_add_module(xstar_cpp bindings/python_bindings.cpp)
target_link_libraries(xstar_cpp PRIVATE xstar_cpp_core)
target_include_directories(xstar_cpp PRIVATE xstar_cpp/include)

# .so 输出到 xstar_cpp_py/ 包目录
set_target_properties(xstar_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/xstar_cpp_py)
