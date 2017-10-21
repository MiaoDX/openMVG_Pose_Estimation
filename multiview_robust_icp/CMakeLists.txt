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


# directories
set (LIBICP_SRC_DIR libicp/src)
# include directory
include_directories("${LIBICP_SRC_DIR}")

# sources
FILE(GLOB LIBICP_SRC_FILES ${LIBICP_SRC_DIR}/icp.cpp ${LIBICP_SRC_DIR}/icpPointToPlane.cpp ${LIBICP_SRC_DIR}/icpPointToPoint.cpp ${LIBICP_SRC_DIR}/kdtree.cpp ${LIBICP_SRC_DIR}/matrix.cpp)

# make release version
#set(CMAKE_BUILD_TYPE Release)

# build demo program
# add_executable(icp ${LIBICP_SRC_FILES})






add_executable(openMVG_sample_multiview_robustICP robust_icp.cpp ${LIBICP_SRC_FILES})
target_link_libraries(openMVG_sample_multiview_robustICP
  openMVG_image
  openMVG_multiview
  openMVG_features
  openMVG_matching
  stlplus)
target_compile_definitions(openMVG_sample_multiview_robustICP
  PRIVATE -DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET openMVG_sample_multiview_robustICP PROPERTY FOLDER OpenMVG/Samples)
