set(TARGET "cuda_kernels")

file(GLOB_RECURSE SOURCES "kernels/*.cu" "*.cc")

cuda_add_library(${TARGET} SHARED ${SOURCES})

set_target_properties(
  ${TARGET}
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             CUDA_EXTENSIONS OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(
    ${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Werror,-Wall -rdc=true
                     -std=c++17 -fconcepts -fpermissive>)

target_compile_features(${TARGET} PUBLIC cxx_std_17 cuda_std_17)

target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")