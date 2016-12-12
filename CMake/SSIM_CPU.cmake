PROJECT(SSIMCPU CXX)
ADD_DEFINITIONS( -D__CPU__ )
ADD_DEFINITIONS( -D__iSOSIA__ )


# Debug settings
IF (DEBUG EQUAL 1)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -g")
set(EXEC_NAME "${EXEC_NAME}_DEBUG")
ELSE()
# Non-debug
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -O3 -fopenmp -march=native -pedantic")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -fopenmp -D_OPENMP -ffast-math -Wall")
ENDIF(DEBUG EQUAL 1)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -fno-unsafe-math-optimizations -march=native -mfpmath=sse -Wall")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -pg")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-finite-math-only -fmath-errno -ftrapping-math -fno-rounding-math -fsignaling-nans -fno-unsafe-math-optimizations")

SET(SSIM_CPU_SOURCE ${SSIM_SOURCE}/Log.cpp ${SSIM_SOURCE}/Error.cpp ${SSIM_SOURCE}/MultigridFunctions.cpp ${SSIM_SOURCE}/iSOSIA.cpp ${SSIM_SOURCE}/multigrid.cpp)
#ADD_LIBRARY(SSIMCPU ${SSIM_CPU_SOURCE} )

# BUILD SSIM
     ADD_EXECUTABLE(${EXEC_NAME} ${SSIM_CPU_SOURCE} ${SSIM_MAIN}.cpp ${BIN})
#TARGET_LINK_LIBRARIES(${EXEC_NAME} SSIMCPU)

# BUILD unit test
#INCLUDE_DIRECTORIES( ${SSIM_SOURCE} )
#FIND_PACKAGE(Boost COMPONENTS unit_test_framework REQUIRED)
#ADD_EXECUTABLE(Unit_Test ${TEST}test_multigrid.cpp ${BIN})
#TARGET_LINK_LIBRARIES(Unit_Test SSIMCPU netcdf)
#TARGET_LINK_LIBRARIES(Unit_Test boost_unit_test_framework)

# Tests
#ADD_TEST(MULTIGRID_TEST ${BIN}Unit_Test)

