set(MATLIB_CMAKE_PATH ${CMAKE_CURRENT_LIST_DIR} CACHE STRING "Matlib installation path")
message("Loading matlib lists from ${MATLIB_CMAKE_PATH}")

set(RISCV_DIR $ENV{RISCV})
set(CMAKE_CXX_COMPILER_FORCED TRUE)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_PK "Build for use with the RISCV Proxy Kernel" OFF)
option(USE_CPU "Use CPU implementation" OFF)
option(USE_GEMMINI "Use Gemmini implementation" OFF)
option(USE_RVV "Use RISCV vector extension implementation" ON)
option(USE_RVVU "Use RISCV vector extension unrolling optimizations" OFF)
option(USE_RVVF "Use RISCV vector extension fused unrolling optimizations" OFF)
option(USE_RVA "Use RISCV vector extension array implementation" OFF)
option(USE_MATVEC "Use matvec instead of matmul for matrix multiplication" ON)
option(MEMORY "Use Memory Optimizations" ON)
option(CHECKSUM "Use checksum to debug component functions" OFF)
option(OPTIMIZED "Use Extra Optimizations" ON)
option(MEASURE_CYCLES "Measure Cycles" OFF)

set(USE_LMUL "1" CACHE STRING "LMUL to use in vector instructions")
add_compile_definitions(USE_LMUL=${USE_LMUL})
set(USE_TYPE "float32" CACHE STRING "Matlib data type")
message("Setting default operand type as ${USE_TYPE}")
add_compile_definitions(USE_TYPE=${USE_TYPE})
set(USE_BATCH "1" CACHE STRING "Innermost loop batch size from 1 up to 8")
message("Setting default innermost loop batch size as ${USE_BATCH}")
add_compile_definitions(BATCH=${USE_BATCH})

if(USE_PK)
    add_compile_definitions(USE_PK=1)
endif(USE_PK)
if(USE_CPU)
    add_compile_definitions(USE_CPU=1)
endif(USE_CPU)
if(USE_GEMMINI)
    add_compile_definitions(USE_GEMMINI=1)
endif(USE_GEMMINI)
if(USE_RVV)
    add_compile_definitions(USE_RVV=1)
endif(USE_RVV)
if(USE_RVVU OR USE_RVVF)
    add_compile_definitions(USE_RVVU=1)
endif(USE_RVVU OR USE_RVVF)
if(USE_RVA)
    add_compile_definitions(USE_RVA=1)
endif(USE_RVA)
if(USE_MATVEC)
    add_compile_definitions(USE_MATVEC=1)
endif(USE_MATVEC)
if(MEMORY)
    add_compile_definitions(MEMORY=1)
endif(MEMORY)
if(CHECKSUM)
    add_compile_definitions(TRACE_CHECKSUMS=1)
endif(CHECKSUM)
if(OPTIMIZED)
    add_compile_definitions(OPTIMIZED=1)
endif(OPTIMIZED)
if(MEASURE_CYCLES)
    add_compile_definitions(MEASURE_CYCLES=1)
endif(MEASURE_CYCLES)

# Add optimization flags
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -Og")

set(STATIC_LINKING TRUE)

if(USE_RVV OR USE_RVVU OR USE_RVVF OR USE_CPU)
    set(WRAP_SPECS_FILE "${RISCV_DIR}/riscv64-unknown-elf/lib/htif_wrap.specs")
    set(SPECS_FILE "${RISCV_DIR}/riscv64-unknown-elf/lib/htif_nano.specs")
    set(LIBGLOSS_DIR "${RISCV_DIR}/riscv64-unknown-elf/lib/")
    if(USE_PK)
        set(CMAKE_CXX_COMPILER ${RISCV_DIR}/bin/riscv64-unknown-linux-gnu-g++)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -fno-builtin-printf")
        set(CMAKE_EXE_LINKER_FLAGS "-static -lm -lstdc++")
    else()
        set(CMAKE_CXX_COMPILER ${RISCV_DIR}/bin/riscv64-unknown-elf-g++)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medany -march=rv64gcv_zfh -mabi=lp64d -fno-common -fno-builtin-printf")
        set(CMAKE_EXE_LINKER_FLAGS "-static -lm -lstdc++ -Wl,-Map=output.map -L${LIBGLOSS_DIR} -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE} -T ${CMAKE_SOURCE_DIR}/include/htif.ld")
        set(SPECS_FILE "${RISCV_DIR}/riscv64-unknown-elf/lib/htif.specs")
    endif()
else()
    set(CMAKE_CXX_COMPILER g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "-static -lm -lstdc++")
endif()

