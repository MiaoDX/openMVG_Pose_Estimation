cmake_minimum_required(VERSION 2.8)
project (openMVG_Sample)


# User options
set( DEBUG_MODE FALSE )
#set( DEBUG_MODE TRUE ) # comment this to use OpenCV 2.x


set (openMVG_DIR C:/tools/cmake_install_libs/openMVG_DEVELOP)

#find_package(OpenMVG REQUIRED)
message(STATUS "DEBUG_MODE value:${DEBUG_MODE}")
if (DEBUG_MODE)
message(STATUS "Using Debug mode ...")

include (${openMVG_DIR}/debug/share/openMVG/cmake/OpenMVGConfig.cmake)
set(CERES_LIBRARIES ${openMVG_DIR}/debug/lib/ceres-debug.lib)
set(EIGEN_DIR ${openMVG_DIR}/debug/third_party/eigen)

else()
message(STATUS "Using Release mode ...")

include (${openMVG_DIR}/release/share/openMVG/cmake/OpenMVGConfig.cmake)
set(CERES_LIBRARIES ${openMVG_DIR}/release/lib/ceres.lib)
set(EIGEN_DIR ${openMVG_DIR}/release/third_party/eigen)
endif() # (DEBUG_MODE) 



include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake )
#add_definitions(-D_CV_VERSION_3) # `#ifdef _CV_VERSION_3`

message(STATUS "OPENMVG_INCLUDE_DIRS:${OPENMVG_INCLUDE_DIRS}")
message(STATUS "OPENMVG_LIBRARIES:${OPENMVG_LIBRARIES}")
message(STATUS "CERES_LIBRARIES:${CERES_LIBRARIES}")
message(STATUS "OpenCV_INCLUDE_DIRS:${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS:${OpenCV_LIBS}")

include_directories(${OPENMVG_INCLUDE_DIRS} ${EIGEN_DIR} ${OpenCV_INCLUDE_DIRS})
link_libraries(${OPENMVG_LIBRARIES} ${OpenCV_LIBS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils) # for the json.hpp



# directories
#FIND_PATH(BOOST_DIR "boost")
set(BOOST_ROOT "C:/tools/boost_1_65_0")
find_package( Boost REQUIRED )

# include directory
include_directories(${Boost_INCLUDE_DIR})

# determine if OpenMP can/should be used
option(USE_OPENMP "Enable OpenMP?" OFF) # set to OFF to disable
if(USE_OPENMP)
 FIND_PACKAGE(OpenMP)
 if(OPENMP_FOUND OR OpenMP_FOUND)
  message(STATUS "OpenMP flags = ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 else(OPENMP_FOUND OR OpenMP_FOUND)
  message(STATUS "OpenMP not supported")
 endif(OPENMP_FOUND OR OpenMP_FOUND)
endif(USE_OPENMP)


add_subdirectory(multiview_robust_homography)
add_subdirectory(multiview_robust_essential)
add_subdirectory(multiview_robust_essential_ba)


add_subdirectory(multiview_robust_essential_cv)

add_subdirectory(multiview_robust_pnp)
add_subdirectory(multiview_robust_pnp_mvg)
add_subdirectory(multiview_robust_pnp_noise)

add_subdirectory(multiview_robust_icp)

add_subdirectory(main_ComputeFeatures_OpenCV)

add_subdirectory(utils)




