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


#include "icpPointToPoint.h"
#include <icpPointToPlane.h>

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

void ICP_With_Point (const vector<cv::Point3f>& pts1, const vector<cv::Point3f>& pts2, vector<FLOAT>& R_vec, vector<FLOAT>& t_vec );

int main( int argc, char **argv ) {

    string im1 = "1.jpg";
    string im2 = "2.jpg";
    string im1_depth = "1_depth.jpg";
    string im2_depth = "2_depth.jpg";
    string output_json_file = "output.json";
    string camera_K_file = "H:/projects/SLAM/dataset/K.txt";

    CmdLine cmd;
    cmd.add ( make_option ( 'a', im1, "image1" ) );
    cmd.add ( make_option ( 'b', im2, "image2" ) );
    cmd.add ( make_option ( 'l', im1_depth, "image1_depth" ) );
    cmd.add ( make_option ( 'r', im2_depth, "image2_depth" ) );
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
            << "[-r|--image2_depth - the name of image2 depth]\n"
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
    cv::Mat d2 = cv::imread ( im2_depth, CV_LOAD_IMAGE_UNCHANGED );
    vector<cv::Point3f> pts_3d_1, pts_3d_2;
    /*
    for ( DMatch m:matches )
    {
        ushort d1 = depth1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) ) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short> ( int ( keypoints_2[m.trainIdx].pt.y ) ) [ int ( keypoints_2[m.trainIdx].pt.x ) ];
        if ( d1==0 || d2==0 )   // bad depth
            continue;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );
        float dd1 = float ( d1 ) /1000.0;
        float dd2 = float ( d2 ) /1000.0;
        pts1.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );
        pts2.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
    }
    */

    cv::Mat K_OCV;
    cv::eigen2cv ( K, K_OCV );
    cout << "K_eigen: " << K << endl;
    cout << "K_OCV: " << K_OCV << endl;


    for ( size_t k = 0; k < xL.cols(); ++k ) {
        
        cv::Point2d kp1 ( xL.col ( k ).x (), xL.col ( k ).y () );
        cv::Point2d kp2 ( xR.col ( k ).x (), xR.col ( k ).y () );

        ushort d_1 = d1.ptr<unsigned short> ( int ( kp1.y ) )[int ( kp1.x )];
        ushort d_2 = d2.ptr<unsigned short> ( int ( kp2.y ) )[int ( kp2.x )];
        if ( d_1 == 0 || d_2 == 0 )   // bad depth
            continue;
        float dd_1 = d_1 / 1000.0;
        float dd_2 = d_2 / 1000.0;

        cv::Point2d p1 = pixel2cam ( kp1, K_OCV );
        cv::Point2d p2 = pixel2cam ( kp2, K_OCV );
        pts_3d_1.push_back ( cv::Point3f ( p1.x*dd_1, p1.y*dd_1, dd_1 ) );
        pts_3d_2.push_back ( cv::Point3f ( p2.x*dd_2, p2.y*dd_2, dd_2 ) );        
    }


    std::vector<double> rotation_vec ( 9 );
    std::vector<double> translation_vec ( 3 );
    //ICP_With_Point ( vector<cv::Point3f> pts1, vector<cv::Point3f> pts2 )
    ICP_With_Point ( pts_3d_1, pts_3d_2, rotation_vec, translation_vec );
    


    // Save R,t to file
    json j;
    j["im1"] = im1;
    j["im2"] = im2;

    std::vector<double> K_vec ( 9 );
    Eigen::Map<Eigen::MatrixXd> ( K_vec.data (), 3, 3 ) = K.transpose ();

    j["R"] = rotation_vec;
    j["t"] = translation_vec;
    j["K"] = K_vec;

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

void ICP_With_Point ( const vector<cv::Point3f>& pts1, const vector<cv::Point3f>& pts2, vector<FLOAT>& R_vec, vector<FLOAT>& t_vec )
{
    int32_t dim = 3;
    //int32_t num = 10000;
    int32_t num = pts1.size ();

    // allocate model and template memory
    double* M = (double*)calloc ( 3 * num, sizeof ( double ) );
    double* T = (double*)calloc ( 3 * num, sizeof ( double ) );

    // set model and template points
    cout << endl << "Creating model with " << num << " points ..." << endl;

    for ( int32_t k = 0; k < pts1.size (); k++ )
    {
        cv::Point3f p1 = pts1[k];
        cv::Point3f p2 = pts2[k];
        //cout << "p1:" << p1 << endl;

        M[k * 3 + 0] = p1.x;
        M[k * 3 + 1] = p1.y;
        M[k * 3 + 2] = p1.z;

        T[k * 3 + 0] = p2.x;
        T[k * 3 + 1] = p2.y;
        T[k * 3 + 2] = p2.z;
    }

    cout << "assign values done" << endl;


    // start with identity as initial transformation
    // in practice you might want to use some kind of prediction here
    Matrix R = Matrix::eye ( 3 );
    Matrix t ( 3, 1 );

    // run point-to-plane ICP (-1 = no outlier threshold)
    cout << endl << "Running ICP" << endl;
    IcpPointToPoint icp ( M, num, dim );
    //IcpPointToPlane icp ( M, num, dim );
    icp.fit ( T, num, R, t, -1 );

    // free memory
    free ( M );
    free ( T );

    // results
    cout << endl << "Transformation results:" << endl;
    cout << "R:" << endl << R << endl << endl;
    cout << "t:" << endl << t << endl << endl;

    
    R.getData ( R_vec.data () );
    t.getData ( t_vec.data () );
    //cout << "R_vec:" << R_vec.data() << endl;
    cout << "R_vec:" << endl;
    for(auto i :R_vec )
    {
        cout << " " << i;
    }
    cout << endl;
    
    cout << "t_vec:" << endl;
    for ( auto i : t_vec )
    {
        cout << " " << i;
    }
    cout << endl;


}