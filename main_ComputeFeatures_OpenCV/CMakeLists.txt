
#add_definitions(-DUSE_OCVSIFT)


add_executable(main_ComputeFeatures_OpenCV main_ComputeFeatures_OpenCV.cpp)
target_link_libraries(main_ComputeFeatures_OpenCV
  openMVG_image
  openMVG_features
  openMVG_matching
  openMVG_system
  openMVG_sfm
  stlplus)
target_compile_definitions(main_ComputeFeatures_OpenCV
  PRIVATE -DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET main_ComputeFeatures_OpenCV PROPERTY FOLDER OpenMVG/Samples)
