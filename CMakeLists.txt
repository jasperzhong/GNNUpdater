cmake_minimum_required(VERSION 3.18)
project(libgnnupdater CXX)
enable_language(CUDA)

set(TARGET_LIB "libgnnupdater")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# set cuda arch
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
message(STATUS "CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")

# 3rd party

# Get Python paths dynamically
find_package(PythonInterp REQUIRED)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_paths as gp; paths = gp(); print(paths['include'], paths['stdlib'])"
    OUTPUT_VARIABLE PYTHON_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCHALL "[^ ]+" PYTHON_PATH_LIST ${PYTHON_PATHS})
list(GET PYTHON_PATH_LIST 0 PYTHON_INCLUDE_DIRS)
list(GET PYTHON_PATH_LIST 1 PYTHON_LIBRARIES)

message(STATUS "Detected Python include dir: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Detected Python library dir: ${PYTHON_LIBRARIES}")

include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})

add_subdirectory(third_party/pybind11)


include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/gnnupdater/csrc)
include_directories(/usr/local/cuda/include)
include_directories(third_party/pybind11/include)
include_directories(third_party/spdlog/include)
include_directories(third_party/rmm/include)


file(GLOB_RECURSE GNNUPDATER_SRC_FILES ${PROJECT_SOURCE_DIR}/gnnupdater/csrc/*.cc)
set_source_files_properties(${PROJECT_SOURCE_DIR}/gnnupdater/csrc/api.cc PROPERTIES LANGUAGE CUDA)
file(GLOB_RECURSE GNNUPDATER_SRC_CUDA_FILES ${PROJECT_SOURCE_DIR}/gnnupdater/csrc/*.cu)
set_source_files_properties(${GNNUPDATER_SRC_CUDA_FILES} PROPERTIES LANGUAGE CUDA)
list(APPEND GNNUPDATER_SRC_FILES ${GNNUPDATER_SRC_CUDA_FILES})

pybind11_add_module(${TARGET_LIB} ${GNNUPDATER_SRC_FILES})

add_subdirectory(third_party/abseil-cpp)
target_link_libraries(${TARGET_LIB} PRIVATE absl::flat_hash_map)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(${TARGET_LIB} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

target_compile_options(${TARGET_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    -rdc=true
    -fopenmp
    >)

set_property(TARGET ${TARGET_LIB} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp -fPIC -Wall -ftree-vectorize")
set(ARCH_FLAGS "-march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
