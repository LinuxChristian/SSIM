# This CMake file compiles the GPU
# version of SSIM
PROJECT(SSIMGPU)
FIND_PACKAGE(CUDA)

INCLUDE(FindCUDA REQUIRED )
IF (OMP EQUAL 1)
   FIND_PACKAGE(OpenMP REQUIRED )
ENDIF (OMP EQUAL 1)

SET(CUDA_SDK_ROOT_DIR "/usr/local/cuda-5.0/sample/common/inc")
# -lm -O3 -fopenmp -ffast-math -pedantic -Wall

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lm;-D__iSOSIA__;-D__GPU__;")
#-Xcompiler;-fopenmp;-lgop

IF (GENERATION EQUAL 1)
# Kepler
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35;-gencode=arch=compute_35,code=compute_35;-fmad=false;-O3")
ELSE()
# Fermi
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_20,code=sm_20;--fmad=false;-O3")
ENDIF (GENERATION EQUAL 1)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DSSIM")

IF (OMP EQUAL 1)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DSSIM;-Xcompiler;-fopenmp;-lgomp;-D__OPENMP;")
#-Xcompiler;-fopenmp;-lgomp;-D__OPENMP;
ENDIF (OMP EQUAL 1)

# Debug
IF (DEBUG EQUAL 1)
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G;-g")
   set(EXEC_NAME "${EXEC_NAME}_DEBUG")
ENDIF(DEBUG EQUAL 1)

IF (BC EQUAL 1)
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D__BC__")
   set(CUDA_NVCC_FLAGS "${CMAKE_CXX_FLAGS};-D__BC__")
ENDIF(BC EQUAL 1)

SET(SSIM_GPU_SOURCE ${SSIM_SOURCE}/Log.cpp ${SSIM_SOURCE}/Error.cu ${SSIM_SOURCE}/MultigridFunctions.cu ${SSIM_SOURCE}/iSOSIA.cu ${SSIM_SOURCE}/multigrid.cu)
CUDA_ADD_LIBRARY(SSIMGPU ${SSIM_GPU_SOURCE} )


# Set GCC Compile flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -DSSIM -ffast-math -D__iSOSIA__ -D__CPU__-O3")

#IF (OMP EQUAL 1)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D_OPENMP")
#ENDIF (OMP EQUAL 1)

# BUILD SSIM
CUDA_ADD_EXECUTABLE(${EXEC_NAME} ${SSIM_MAIN}.cpp ${BIN})
TARGET_LINK_LIBRARIES(${EXEC_NAME} SSIMGPU)

IF (OMP EQUAL 1)
   TARGET_LINK_LIBRARIES(${EXEC_NAME} gomp)
ENDIF (OMP EQUAL 1)
