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

#add_subdirectory(cameras_undisto_Brown)
#add_subdirectory(multiview_robust_estimation_tutorial)
add_subdirectory(multiview_robust_homography)
#add_subdirectory(multiview_robust_homography_guided)
#add_subdirectory(multiview_robust_fundamental)
#add_subdirectory(multiview_robust_fundamental_guided)
add_subdirectory(multiview_robust_essential)
#add_subdirectory(multiview_robust_essential_spherical)
add_subdirectory(multiview_robust_essential_ba)

add_subdirectory(multiview_robust_pnp)


#
#add_subdirectory(exif_Parsing)
#
#add_subdirectory(features_repeatability)
#add_subdirectory(features_affine_demo)
#add_subdirectory(features_kvld_filter)
#add_subdirectory(features_siftPutativeMatches)
#add_subdirectory(features_image_matching)
#add_subdirectory(describe_and_match_GUI)
#
##add_subdirectory(geodesy_show_exif_gps_position) # needs helper function in source code
#add_subdirectory(image_spherical_to_pinholes)
#
#add_subdirectory(image_undistort_gui)



add_subdirectory(main_ComputeFeatures_OpenCV)
