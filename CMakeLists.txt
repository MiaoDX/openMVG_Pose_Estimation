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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # for the json.hpp


add_subdirectory(multiview_robust_homography)
add_subdirectory(multiview_robust_essential)
#add_subdirectory(multiview_robust_essential_ba)

add_subdirectory(multiview_robust_pnp)
add_subdirectory(multiview_robust_pnp_mvg)
#add_subdirectory(multiview_robust_icp)

add_subdirectory(main_ComputeFeatures_OpenCV)
