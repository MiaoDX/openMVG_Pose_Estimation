
add_executable(openMVG_sample_multiview_robustPnPMVG robust_pnp_mvg.cpp)
target_link_libraries(openMVG_sample_multiview_robustPnPMVG
  openMVG_image
  openMVG_multiview
  openMVG_features
  openMVG_matching
  stlplus)
target_compile_definitions(openMVG_sample_multiview_robustPnPMVG
  PRIVATE -DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET openMVG_sample_multiview_robustPnPMVG PROPERTY FOLDER OpenMVG/Samples)
