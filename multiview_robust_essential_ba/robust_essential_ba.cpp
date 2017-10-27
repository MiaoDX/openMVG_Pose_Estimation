// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/cameras/Camera_Pinhole.hpp"
#include "openMVG/cameras/Camera_Pinhole_Radial.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "openMVG/features/svg_features.hpp"
#include "openMVG/geometry/pose3.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/image/image_concat.hpp"
#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/matching/svg_matches.hpp"
#include "openMVG/multiview/triangulation.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/pipelines/sfm_robust_model_estimation.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_BA.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <cmdLine/cmdLine.h>

#include "json.hpp"

#include "robustRelativePoseHelper.h"

#include "AFD.h"

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::sfm;
using namespace std;

using json = nlohmann::json;

/// Read intrinsic K matrix from a file (ASCII)
/// F 0 ppx
/// 0 F ppy
/// 0 0 1
bool readIntrinsic(const std::string & fileName, Mat3 & K);

openMVG::geometry::Pose3 relative_pose( const openMVG::geometry::Pose3 query_pose, const openMVG::geometry::Pose3 reference_pose)
{
    const openMVG::geometry::Pose3 relativePose = query_pose * (reference_pose.inverse ());
    Eigen::MatrixXd R_mvg = relativePose.rotation ();
    Eigen::MatrixXd t_mvg = relativePose.translation ();
    cout << "SFM: R:\n" << R_mvg << endl;
    cout << "SFM: t:\n" << t_mvg << endl;
    return relativePose;
}


void save_Rt_to_json( openMVG::geometry::Pose3 relativePose, json& j)
{
    Eigen::MatrixXd R_mvg = relativePose.rotation ();
    Eigen::MatrixXd t_mvg = relativePose.translation ();

    
    std::vector<double> rotation_vec ( 9 );
    Eigen::Map<Eigen::MatrixXd> ( rotation_vec.data (), 3, 3 ) = relativePose.rotation ().transpose ();
    Vec3 translation_Vec = relativePose.translation ();
    std::vector<double> translation_vec{ translation_Vec ( 0 ) , translation_Vec ( 1 ), translation_Vec ( 2 ) };
    
    cout << "In save_Rt_to_json:" << endl;
    cout << "    R:\n" << R_mvg << endl;
    cout << "    t:\n" << t_mvg << endl;
    

    j["R"] = rotation_vec;
    j["t"] = translation_vec;

    
}




/// Show :
///  how computing an essential with know internal calibration matrix K
///  how refine the camera motion, focal and structure with Bundle Adjustment
///   way 1: independent cameras [R|t|f] and structure
///   way 2: independent cameras motion [R|t], shared focal [f] and structure
int main( int argc, char **argv ) 
{

  //const std::string sInputDir = stlplus::folder_up(string(THIS_SOURCE_DIR))
  //  + "/imageData/SceauxCastle/";
  ////Image<RGBColor> image;
  //const string jpg_filenameL = sInputDir + "100_7101.jpg";
  //const string jpg_filenameR = sInputDir + "100_7102.jpg";

    
    string camera_K_file = "H:/projects/SLAM/dataset/K.txt";
    string im1 = "1.jpg";
    string im2 = "2.jpg";
    string output_json_file = "output.json";

    CmdLine cmd;
    cmd.add ( make_option ( 'K', camera_K_file, "camera_K_file" ) );
    cmd.add ( make_option ( 'a', im1, "image1" ) );
    cmd.add ( make_option ( 'b', im2, "image2" ) );
    cmd.add ( make_option ( 'o', output_json_file, "output_json file" ) );


    try {
        if ( argc == 1 ) throw std::string ( "Invalid command line parameter." );
        cmd.process ( argc, argv );
    }
    catch ( const std::string& s ) {
        std::cerr << "Usage: " << argv[0] << ' '
            << "[-K|--camera_K_file - the file stores K values]\n"
            << "[-a|--image1 - the file name of image1, absolute path, eg. H:/dataset/1.jpg]\n"
            << "[-b|--image2 - the name of image2]\n"
            << "[-o|--output_json_file - json file for the R,t]\n"
            << std::endl;

        std::cerr << s << std::endl;
        return EXIT_FAILURE;
    }






    const string jpg_filenameL = im1;
    const string jpg_filenameR = im2;

    const string save_path = im1 + '_' + im2;
    stlplus::folder_create ( save_path );
    stlplus::folder_set_current ( save_path );


  Image<unsigned char> imageL, imageR;
  ReadImage(jpg_filenameL.c_str(), &imageL);
  ReadImage(jpg_filenameR.c_str(), &imageR);

  //--
  // Detect regions thanks to an image_describer
  //--
  using namespace openMVG::features;
  std::unique_ptr<Image_describer> image_describer(new SIFT_Anatomy_Image_describer);
  std::map<IndexT, std::unique_ptr<features::Regions> > regions_perImage;
  image_describer->Describe(imageL, regions_perImage[0]);
  image_describer->Describe(imageR, regions_perImage[1]);

  const SIFT_Regions* regionsL = dynamic_cast<SIFT_Regions*>(regions_perImage.at(0).get());
  const SIFT_Regions* regionsR = dynamic_cast<SIFT_Regions*>(regions_perImage.at(1).get());

  const PointFeatures
    featsL = regions_perImage.at(0)->GetRegionsPositions(),
    featsR = regions_perImage.at(1)->GetRegionsPositions();

  // Show both images side by side
  {
    Image<unsigned char> concat;
    ConcatH(imageL, imageR, concat);
    string out_filename = "01_concat.jpg";
    WriteImage(out_filename.c_str(), concat);
  }

  //- Draw features on the two image (side by side)
  {
    Features2SVG
    (
      jpg_filenameL,
      {imageL.Width(), imageL.Height()},
      regionsL->Features(),
      jpg_filenameR,
      {imageR.Width(), imageR.Height()},
      regionsR->Features(),
      "02_features.svg"
    );
  }

  std::vector<IndMatch> vec_PutativeMatches;
  //-- Perform matching -> find Nearest neighbor, filtered with Distance ratio
  {
    // Find corresponding points
    matching::DistanceRatioMatch(
      0.8, matching::BRUTE_FORCE_L2,
      *regions_perImage.at(0).get(),
      *regions_perImage.at(1).get(),
      vec_PutativeMatches);

    IndMatchDecorator<float> matchDeduplicator(
            vec_PutativeMatches, featsL, featsR);
    matchDeduplicator.getDeduplicated(vec_PutativeMatches);

    std::cout
      << regions_perImage.at(0)->RegionCount() << " #Features on image A" << std::endl
      << regions_perImage.at(1)->RegionCount() << " #Features on image B" << std::endl
      << vec_PutativeMatches.size() << " #matches with Distance Ratio filter" << std::endl;

    // Draw correspondences after Nearest Neighbor ratio filter
    const bool bVertical = true;
    Matches2SVG
    (
      jpg_filenameL,
      {imageL.Width(), imageL.Height()},
      regionsL->GetRegionsPositions(),
      jpg_filenameR,
      {imageR.Width(), imageR.Height()},
      regionsR->GetRegionsPositions(),
      vec_PutativeMatches,
      "03_Matches.svg",
      bVertical
    );
  }

  // Essential geometry filtering of putative matches
  {
    Mat3 K;
    //read K from file
    if (!readIntrinsic(camera_K_file, K))
    {
      std::cerr << "Cannot read intrinsic parameters." << std::endl;
      return EXIT_FAILURE;
    }

    //A. prepare the corresponding putatives points
    Mat xL(2, vec_PutativeMatches.size());
    Mat xR(2, vec_PutativeMatches.size());
    for (size_t k = 0; k < vec_PutativeMatches.size(); ++k)  {
      const PointFeature & imaL = featsL[vec_PutativeMatches[k].i_];
      const PointFeature & imaR = featsR[vec_PutativeMatches[k].j_];
      xL.col(k) = imaL.coords().cast<double>();
      xR.col(k) = imaR.coords().cast<double>();
    }

    //B. Compute the relative pose thanks to a essential matrix estimation
    std::pair<size_t, size_t> size_imaL(imageL.Width(), imageL.Height());
    std::pair<size_t, size_t> size_imaR(imageR.Width(), imageR.Height());
    RelativePose_Info relativePose_info;
    //if (!robustRelativePose(K, K, xL, xR, relativePose_info, size_imaL, size_imaR, 4096 ))
    if ( !robustRelativePoseCopy ( K, K, xL, xR, relativePose_info, size_imaL, size_imaR, 4096 ) )
    {
      std::cerr << " /!\\ Robust relative pose estimation failure."
        << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "\nFound an Essential matrix:\n"
      << "\tprecision: " << relativePose_info.found_residual_precision << " pixels\n"
      << "\t#inliers: " << relativePose_info.vec_inliers.size() << "\n"
      << "\t#matches: " << vec_PutativeMatches.size()
      << std::endl;

    // Show Essential validated point
    const bool bVertical = true;
    InlierMatches2SVG
    (
      jpg_filenameL,
      {imageL.Width(), imageL.Height()},
      regionsL->GetRegionsPositions(),
      jpg_filenameR,
      {imageR.Width(), imageR.Height()},
      regionsR->GetRegionsPositions(),
      vec_PutativeMatches,
      relativePose_info.vec_inliers,
      "04_ACRansacEssential.svg",
      bVertical
    );

    std::cout << std::endl
      << "-- Rotation|Translation matrices: --" << "\n"
      << relativePose_info.relativePose.rotation() << "\n\n"
      << relativePose_info.relativePose.translation() << "\n" << std::endl;

    // Save R,t to file
    json j;
    j["im1"] = im1;
    j["im2"] = im2;

    std::vector<double> K_vec ( 9 );
    Eigen::Map<Eigen::MatrixXd> ( K_vec.data (), 3, 3 ) = K.transpose ();
    j["K"] = K_vec;
    

    

    /*
    save_Rt_to_json ( relativePose_info.relativePose, j );
    // write prettified JSON to another file
    cout << "Going to save json to " << output_json_file << endl;
    std::ofstream o ( output_json_file );
    o << std::setw ( 4 ) << j << std::endl;
    cout << "Save json done" << endl;
    */

    //C. Triangulate and check valid points
    // invalid points that do not respect cheirality are discarded (removed
    //  from the list of inliers).

    cout << "BA with Refine [X],[R|t], shared [f, ppx, ppy]" << endl;
    //std::cout << "Which BA do you want ?\n"
    //  << "\t 1: Refine [X],[f,ppx,ppy,R|t] (individual cameras)\n"
    //  << "\t 2: Refine [X],[R|t], shared [f, ppx, ppy]\n"
    //  << "\t 3: Refine [X],[R|t], shared brown K3 distortion model [f,ppx,ppy,k1,k2,k3]\n" << std::endl;
    //int iBAType = -1;
    //std::cin >> iBAType;
    int iBAType = 2;
    const bool bSharedIntrinsic = (iBAType == 2 || iBAType == 3) ? true : false;

    // Setup a SfM scene with two view corresponding the pictures
    SfM_Data tiny_scene;
    tiny_scene.views[0].reset(new View("", 0, bSharedIntrinsic ? 0 : 1, 0, imageL.Width(), imageL.Height()));
    tiny_scene.views[1].reset(new View("", 1, bSharedIntrinsic ? 0 : 1, 1, imageR.Width(), imageR.Height()));
    // Setup intrinsics camera data
    switch (iBAType)
    {
      case 1: // Each view use it's own pinhole camera intrinsic
        tiny_scene.intrinsics[0].reset(new Pinhole_Intrinsic(imageL.Width(), imageL.Height(), K(0, 0), K(0, 2), K(1, 2)));
        tiny_scene.intrinsics[1].reset(new Pinhole_Intrinsic(imageR.Width(), imageR.Height(), K(0, 0), K(0, 2), K(1, 2)));
        break;
      case 2: // Shared pinhole camera intrinsic
        tiny_scene.intrinsics[0].reset(new Pinhole_Intrinsic(imageL.Width(), imageL.Height(), K(0, 0), K(0, 2), K(1, 2)));
        break;
      case 3: // Shared pinhole camera intrinsic with radial K3 distortion
        tiny_scene.intrinsics[0].reset(new Pinhole_Intrinsic_Radial_K3(imageL.Width(), imageL.Height(), K(0, 0), K(0, 2), K(1, 2)));
        break;
      default:
        std::cerr << "Invalid input number" << std::endl;
        return EXIT_FAILURE;
    }

    // Setup poses camera data
    const Pose3 pose0 = tiny_scene.poses[tiny_scene.views[0]->id_pose] = Pose3(Mat3::Identity(), Vec3::Zero());
    const Pose3 pose1 = tiny_scene.poses[tiny_scene.views[1]->id_pose] = relativePose_info.relativePose;

    std::cout << std::endl
        << "-- Before BA --" << "\n";
    relative_pose ( pose1, pose0 );



    // Init structure by inlier triangulation
    const Mat34 P1 = tiny_scene.intrinsics[tiny_scene.views[0]->id_intrinsic]->get_projective_equivalent(pose0);
    const Mat34 P2 = tiny_scene.intrinsics[tiny_scene.views[1]->id_intrinsic]->get_projective_equivalent(pose1);

    Landmarks & landmarks = tiny_scene.structure;
    for (size_t i = 0; i < relativePose_info.vec_inliers.size(); ++i)  {
      const SIOPointFeature & LL = regionsL->Features()[vec_PutativeMatches[relativePose_info.vec_inliers[i]].i_];
      const SIOPointFeature & RR = regionsR->Features()[vec_PutativeMatches[relativePose_info.vec_inliers[i]].j_];
      // Point triangulation
      Vec3 X;
      TriangulateDLT(
        P1, LL.coords().cast<double>().homogeneous(),
        P2, RR.coords().cast<double>().homogeneous(), &X);
      // Reject point that is behind the camera
      if (pose0.depth(X) < 0 && pose1.depth(X) < 0)
        continue;
      // Add a new landmark (3D point with it's 2d observations)
      landmarks[i].obs[tiny_scene.views[0]->id_view] = Observation(LL.coords().cast<double>(), vec_PutativeMatches[relativePose_info.vec_inliers[i]].i_);
      landmarks[i].obs[tiny_scene.views[1]->id_view] = Observation(RR.coords().cast<double>(), vec_PutativeMatches[relativePose_info.vec_inliers[i]].j_);
      landmarks[i].X = X;
    }
    Save(tiny_scene, "EssentialGeometry_start.ply", ESfM_Data(ALL));

    //D. Perform Bundle Adjustment of the scene

    Bundle_Adjustment_Ceres bundle_adjustment_obj;
    /*bundle_adjustment_obj.Adjust(tiny_scene,
      Optimize_Options(
        Intrinsic_Parameter_Type::ADJUST_ALL,
        Extrinsic_Parameter_Type::ADJUST_ALL,
        Structure_Parameter_Type::ADJUST_ALL));*/

    bundle_adjustment_obj.Adjust ( tiny_scene,
        Optimize_Options (
            Intrinsic_Parameter_Type::NONE,
            Extrinsic_Parameter_Type::ADJUST_ALL,
            Structure_Parameter_Type::NONE ) );

    Save(tiny_scene, "EssentialGeometry_refined.ply", ESfM_Data(ALL));

    Pose3 pose00 = tiny_scene.poses[tiny_scene.views[0]->id_pose];
    Pose3 pose11 = tiny_scene.poses[tiny_scene.views[1]->id_pose];

    std::cout << std::endl
        << "-- After BA --" << "\n";
    const openMVG::geometry::Pose3 relativePose_after_BA = relative_pose ( pose11, pose00 );

    save_Rt_to_json ( relativePose_after_BA, j );
    

    j["AFD"] = AFD ( xL, xR );

    // write prettified JSON to another file
    cout << "Going to save json to " << output_json_file << endl;
    std::ofstream o ( output_json_file );
    o << std::setw ( 4 ) << j << std::endl;
    cout << "Save json done" << endl;



  }




  return EXIT_SUCCESS;
}

bool readIntrinsic(const std::string & fileName, Mat3 & K)
{
  // Load the K matrix
  ifstream in;
  in.open( fileName.c_str(), ifstream::in);
  if (in.is_open())  {
    for (int j=0; j < 3; ++j)
      for (int i=0; i < 3; ++i)
        in >> K(j,i);
  }
  else  {
    std::cerr << std::endl
      << "Invalid input K.txt file" << std::endl;
    return false;
  }
  return true;
}
