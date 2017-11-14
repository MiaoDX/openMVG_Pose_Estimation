add_executable(generate_matched_3d_points generate_matched_3d_points.cpp)
target_link_libraries(generate_matched_3d_points
  openMVG_image
  openMVG_multiview
  openMVG_features
  openMVG_matching
  stlplus)