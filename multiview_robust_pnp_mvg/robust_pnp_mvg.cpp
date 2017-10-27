// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/features/feature.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "openMVG/features/svg_features.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/image/image_concat.hpp"
#include "openMVG/image/image_warping.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/matching/svg_matches.hpp"
#include "openMVG/multiview/solver_homography_kernel.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansac.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansacKernelAdaptator.hpp"
#include "openMVG/types.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include <cmdLine/cmdLine.h>

#include "json.hpp"


/// OpenCV Includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"


#include "openMVG/multiview/solver_resection_kernel.hpp"
#include "openMVG/multiview/solver_resection_p3p.hpp"
#include <openMVG/sfm/sfm.hpp>

#include "AFD.h"

using namespace openMVG::sfm;

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::robust;
using namespace std;


using json = nlohmann::json;


bool readIntrinsic ( const std::string & fileName, Mat3 & K );
cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K )
{
    return cv::Point2d
    (
        (p.x - K.at<double> ( 0, 2 )) / K.at<double> ( 0, 0 ),
        (p.y - K.at<double> ( 1, 2 )) / K.at<double> ( 1, 1 )
    );
}

int main( int argc, char **argv ) {

    string im1 = "1.jpg";
    string im2 = "2.jpg";
    string im1_depth = "1_depth.jpg";
    string output_json_file = "output.json";
    string camera_K_file = "H:/projects/SLAM/dataset/K.txt";

    CmdLine cmd;
    cmd.add ( make_option ( 'a', im1, "image1" ) );
    cmd.add ( make_option ( 'b', im2, "image2" ) );
    cmd.add ( make_option ( 'l', im1_depth, "image1_depth" ) );
    cmd.add ( make_option ( 'K', camera_K_file, "camera_K_file" ) );
    cmd.add ( make_option ( 'o', output_json_file, "output_json file" ) );



    try {
        if ( argc == 1 ) throw std::string ( "Invalid command line parameter." );
        cmd.process ( argc, argv );
    }
    catch ( const std::string& s ) {
        std::cerr << "Usage: " << argv[0] << ' '
            << "[-a|--image1 - the file name of image1, absolute path, eg. H:/dataset/1.jpg]\n"
            << "[-b|--image2 - the name of image2]\n"
            << "[-l|--image1_depth - the name of image1 depth]\n"
            << "[-K|--camera_K_file - the file stores K values]\n"
            << "[-o|--output_json_file - json file for the R,t]\n"
            << std::endl;

        std::cerr << s << std::endl;
        return EXIT_FAILURE;
    }




  Image<RGBColor> image;
  //const string jpg_filenameL = stlplus::folder_up(string(THIS_SOURCE_DIR))
  //  + "/imageData/StanfordMobileVisualSearch/Ace_0.png";
  //const string jpg_filenameR = stlplus::folder_up(string(THIS_SOURCE_DIR))
  //  + "/imageData/StanfordMobileVisualSearch/Ace_0.png";

  //const string jpg_filenameL = "H:/projects/graduation_project_codebase/ACR/dataset/nine_scene/1_000.png";
  //const string jpg_filenameR = "H:/projects/graduation_project_codebase/ACR/dataset/nine_scene/1_500.png";

  const string jpg_filenameL = im1;
  const string jpg_filenameR = im2;

  Image<unsigned char> imageL, imageR;
  ReadImage(jpg_filenameL.c_str(), &imageL);
  ReadImage(jpg_filenameR.c_str(), &imageR);

  //--
  // Detect regions thanks to an image_describer
  //--
  using namespace openMVG::features;
  std::unique_ptr<Image_describer> image_describer
    (new SIFT_Anatomy_Image_describer(SIFT_Anatomy_Image_describer::Params(-1)));
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

    IndMatchDecorator<float> matchDeduplicator (
        vec_PutativeMatches, featsL, featsR );
    matchDeduplicator.getDeduplicated ( vec_PutativeMatches );

    std::cout
        << regions_perImage.at ( 0 )->RegionCount () << " #Features on image A" << std::endl
        << regions_perImage.at ( 1 )->RegionCount () << " #Features on image B" << std::endl
        << vec_PutativeMatches.size () << " #matches with Distance Ratio filter" << std::endl;


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

  // Homography geometry filtering of putative matches
  {
    //A. get back interest point and send it to the robust estimation framework
    Mat xL(2, vec_PutativeMatches.size());
    Mat xR(2, vec_PutativeMatches.size());

    for (size_t k = 0; k < vec_PutativeMatches.size(); ++k)  {
      const PointFeature & imaL = featsL[vec_PutativeMatches[k].i_];
      const PointFeature & imaR = featsR[vec_PutativeMatches[k].j_];
      xL.col(k) = imaL.coords().cast<double>();
      xR.col(k) = imaR.coords().cast<double>();
    }


    Mat3 K;
    //read K from file
    if ( !readIntrinsic ( camera_K_file, K ) )
    {
        std::cerr << "Cannot read intrinsic parameters." << std::endl;
        return EXIT_FAILURE;
    }

    // 建立3D点
    cv::Mat d1 = cv::imread ( im1_depth, CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    /*
    for ( DMatch m : matches )
    {
        ushort d = d1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) )[int ( keypoints_1[m.queryIdx].pt.x )];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d / 1000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }
    */

    cv::Mat K_OCV;
    cv::eigen2cv ( K, K_OCV );
    cout << "K_eigen: " << K << endl;
    cout << "K_OCV: " << K_OCV << endl;


    for ( size_t k = 0; k < xL.cols(); ++k ) {
        
        cv::Point2d kp1 ( xL.col ( k ).x (), xL.col ( k ).y () );
        cv::Point2d kp2 ( xR.col ( k ).x (), xR.col ( k ).y () );

        ushort d = d1.ptr<unsigned short> ( int ( kp1.y ) )[int ( kp1.x )];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d / 1000.0;

        cv::Point2d p1 = pixel2cam ( kp1, K_OCV );
        pts_3d.push_back ( cv::Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( kp2 );
    }


    cv::Mat r, t;
    //solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    solvePnPRansac ( pts_3d, pts_2d, K_OCV, cv::Mat (), r, t, false );
    cv::Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵


    cout << "All matched points num: " << pts_3d.size() << endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;



    Mat pt3D ( 3, pts_3d.size () ), pt2D ( 2, pts_3d.size () );;

    //pt3D = Mat( pts_3d.data ());
    //pt2D = Mat ( pts_2d.data () );

      
    for ( size_t k = 0; k < pts_3d.size (); ++k ) {
        pt3D.col ( k ) = Eigen::Vector3d ( pts_3d[k].x, pts_3d[k].y, pts_3d[k].z ).cast<double> ();
        //pt2D.col ( k ) = Eigen::Vector2d ( pts_2d[k].x, pts_2d[k].y ).cast<double> ();
        pt2D.col ( k ) = Eigen::Vector2d ( pts_2d[k].x, pts_2d[k].y ).cast<double> ();
    }
    
    cout << "pts3D cols:" << pt3D.cols() << ", rows:" << pt3D.rows() << endl;

    geometry::Pose3 pose;
    sfm::Image_Localizer_Match_Data matching_data;

    matching_data.pt3D = pt3D;
    matching_data.pt2D = pt2D;

    sfm::SfM_Localization_Single_3DTrackObservation_Database localizer;

    cameras::Pinhole_Intrinsic_Radial_K3 *ptrPinhole = new cameras::Pinhole_Intrinsic_Radial_K3 (
        imageL.Width (), imageL.Height (), K_OCV.at<double>(0, 0), K_OCV.at<double>(0, 2), K_OCV.at<double>(1, 2), 0., 0., 0. );
    // (w, h, config._fx, config._cx, config._cy, 0., 0., 0.);

    std::shared_ptr<cameras::IntrinsicBase> optional_intrinsic  = std::make_shared<cameras::Pinhole_Intrinsic_Radial_K3> (
            imageL.Width (), imageL.Height (),
            ptrPinhole->focal (), ptrPinhole->principal_point ()[0], ptrPinhole->principal_point ()[1],
            ptrPinhole->getParams ()[3], ptrPinhole->getParams ()[4], ptrPinhole->getParams ()[5] );    
        
    bool bSuccessfulLocalization = false;
    
    // P3P_KE_CVPR17 DLT_6POINTS P3P_KNEIP_CVPR11
    if ( !SfM_Localizer::Localize ( resection::SolverType::P3P_KE_CVPR17, Pair ( imageL.Width (), imageL.Height () ), optional_intrinsic.get (),
        matching_data, pose )
        )
    //if ( !SfM_Localizer::Localize ( Pair ( imageL.Width (), imageL.Height () ), optional_intrinsic.get (),
    //    matching_data, pose )
    //)
    {
        std::cerr << "Cannot locate the image " << im1 << std::endl;
        bSuccessfulLocalization = false;
    }
    else
    {        
        if ( !sfm::SfM_Localizer::RefinePose
        (
            optional_intrinsic.get (),
            pose, matching_data,
            true, optional_intrinsic.get()
        ) )
        {
            std::cerr << "Refining pose for image " << im1 << " failed." << std::endl;
        }

        bSuccessfulLocalization = true;

//        cout << "The refined answers:" << endl;

    }


    // https://github.com/openMVG/openMVG/blob/develop/src/openMVG/sfm/pipelines/sfm_robust_model_estimation.cpp
    // Store [R|C] for the second camera, since the first camera is [Id|0]
    //relativePose_info.relativePose = geometry::Pose3 ( R, -R.transpose () * t );
    Pose3 relativePose = Pose3 ( pose.rotation (), -pose.rotation ().transpose ()*pose.translation () );

    
    cout << "Pose2 rotation:\n" << pose.rotation () << endl;
    cout << "Pose2 translation:\n" << pose.translation () << endl;
    cout << "Pose2 ceter:\n" << pose.center () << endl;


    cout << "Relative Pose rotation:\n" << relativePose.rotation () << endl;
    cout << "Relative Pose translation:\n" << relativePose.translation () << endl;
    cout << "Relative Pose translation ceter:\n" << relativePose.center () << endl;
    
    Vec3 lookingDir = relativePose.rotation ().transpose () * Vec3 ( 0, 0, 1 );
    cout << "Looking at: " << lookingDir << endl;
    Vec3 euler_angles = relativePose.rotation ().eulerAngles ( 2, 1, 0 );
    cout << "euler_angles:" << euler_angles << endl;
    Vec3 euler_angles2 = relativePose.rotation ().eulerAngles ( 0, 1, 2 );
    cout << "euler_angles2:" << euler_angles2 << endl;

    // Save R,t to file
    json j;
    j["im1"] = im1;
    j["im2"] = im2;

    Mat R_eigen;
    std::vector<double> rotation_vec ( 9 );
    Eigen::Map<Eigen::MatrixXd> ( rotation_vec.data (), 3, 3 ) = relativePose.rotation().transpose();
    std::vector<double> translation_vec{ relativePose.translation ()( 0 ) , relativePose.translation () ( 1 ), relativePose.translation () ( 2 ) };
    std::vector<double> K_vec ( 9 );
    Eigen::Map<Eigen::MatrixXd> ( K_vec.data (), 3, 3 ) = K.transpose ();

    j["R"] = rotation_vec;
    j["t"] = translation_vec;
    j["K"] = K_vec;

    j["AFD"] = AFD ( xL, xR );

    // write prettified JSON to another file
    cout << "Going to save json to " << output_json_file << endl;
    std::ofstream o ( output_json_file );
    o << std::setw ( 4 ) << j << std::endl;
    cout << "Save json done" << endl;

  }
  return EXIT_SUCCESS;
}


bool readIntrinsic ( const std::string & fileName, Mat3 & K )
{
    // Load the K matrix
    ifstream in;
    in.open ( fileName.c_str (), ifstream::in );
    if ( in.is_open () ) {
        for ( int j = 0; j < 3; ++j )
            for ( int i = 0; i < 3; ++i )
                in >> K ( j, i );
    }
    else {
        std::cerr << std::endl
            << "Invalid input K.txt file" << std::endl;
        return false;
    }
    return true;
}
