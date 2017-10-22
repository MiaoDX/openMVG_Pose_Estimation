
add_executable(TEST Test.cpp)
target_link_libraries(TEST
  openMVG_image
  openMVG_multiview
  openMVG_system
  openMVG_features
  openMVG_geometry
  openMVG_matching
  openMVG_sfm)