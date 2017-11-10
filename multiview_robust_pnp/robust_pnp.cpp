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

#include "AFD.h"
#include "image_process.h"

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::robust;
using namespace std;


using json = nlohmann::json;

int main( int argc, char **argv ) {

    string im1 = "1.jpg";
    string im2 = "2.jpg";
    string im1_depth = "1_depth.jpg";
    string output_json_file = "output.json";
    string camera_K_file = "H:/projects/SLAM/dataset/K.txt";
    string H_filter = "0";

    CmdLine cmd;
    cmd.add ( make_option ( 'a', im1, "image1" ) );
    cmd.add ( make_option ( 'b', im2, "image2" ) );
    cmd.add ( make_option ( 'l', im1_depth, "image1_depth" ) );
    cmd.add ( make_option ( 'K', camera_K_file, "camera_K_file" ) );
    cmd.add ( make_option ( 'o', output_json_file, "output_json" ) );
    cmd.add ( make_option ( 'H', H_filter, "H_filter" ) );



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
            << "[-H|--H_filter, 0 or 1]\n"
            << std::endl;

        std::cerr << s << std::endl;
        return EXIT_FAILURE;
    }


  const string jpg_filenameL = im1;
  const string jpg_filenameR = im2;

  std::vector<IndMatch> vec_PutativeMatches;
  Mat xL, xR, xL_Hfiltering, xR_Hfiltering;

  get_matches ( jpg_filenameL, jpg_filenameR, vec_PutativeMatches, xL, xR, xL_Hfiltering, xR_Hfiltering );

  cout << "H_filter:" << H_filter << endl;
  if(H_filter == "1" )
  {
      cout << "Using H_filter, replace xL and xR with filtered one" << endl;
      xL = xL_Hfiltering;
      xR = xR_Hfiltering;
  }

  


  // Homography geometry filtering of putative matches
  {
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


    // Save R,t to file
    json j;
    j["im1"] = im1;
    j["im2"] = im2;

    Mat R_eigen;
    cv::cv2eigen ( R, R_eigen );
    std::vector<double> rotation_vec ( 9 );
    Eigen::Map<Eigen::MatrixXd> ( rotation_vec.data (), 3, 3 ) = R_eigen.transpose();
    std::vector<double> translation_vec{ t.at<double>( 0 ) , t.at<double> ( 1 ), t.at<double> ( 2 ) };
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