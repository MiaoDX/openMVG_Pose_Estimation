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

/// OpenCV Includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"

#include "json.hpp"



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
bool readIntrinsic ( const std::string & fileName, Mat3 & K );

vector<double> getVector ( const cv::Mat &_t1f )
{
    cv::Mat t1f;
    _t1f.convertTo ( t1f, CV_64F );
    return (vector<double>)(t1f.reshape ( 1, 1 ));
}

void _calculateRT_CV3_with_ratio (
    const vector<cv::Point2f> points1,
    const vector<cv::Point2f> points2,
    const cv::Mat K,
    cv::Mat& R, cv::Mat& t,
    double& inliers_ratio
    )
{
    assert ( points1.size () > 0 && points1.size () == points2.size () && K.size () == Size ( 3, 3 ) );
    R.release ();
    t.release ();


    //-- 计算本质矩阵
    cv::Mat E = cv::findEssentialMat ( points1, points2, K );

    //cout << "E matrix:\n" << E << endl;

    cv::Mat inliersMask;
    //-- 从本质矩阵中恢复旋转和平移信息.re
    cv::recoverPose ( E, points1, points2, K, R, t, inliersMask );
    // cout << "inliersMask, channels:" << inliersMask.channels () << ", type:" << inliersMask.type () << ", size:" << inliersMask.size() << endl;
    vector<cv::Point2f> inliers_pts1, inliers_pts2;
    for ( int i = 0; i < inliersMask.rows; i++ ) {
        if ( inliersMask.at<uchar> ( i, 0 ) ) {
            inliers_pts1.push_back ( points1[i] );
            inliers_pts2.push_back ( points2[i] );
        }
    }

    inliers_ratio = static_cast<double>(inliers_pts1.size ()) / points1.size ();

    cout << "In recoverPose, points:" << points1.size () << "->" << inliers_pts1.size () << ", ratio:" << inliers_ratio << endl;
}


openMVG::geometry::Pose3 relative_pose ( const openMVG::geometry::Pose3 query_pose, const openMVG::geometry::Pose3 reference_pose )
{
    const openMVG::geometry::Pose3 relativePose = query_pose * (reference_pose.inverse ());
    Eigen::MatrixXd R_mvg = relativePose.rotation ();
    Eigen::MatrixXd t_mvg = relativePose.translation ();
    cout << "SFM: R:\n" << R_mvg << endl;
    cout << "SFM: t:\n" << t_mvg << endl;
    return relativePose;
}


void save_Rt_to_json ( openMVG::geometry::Pose3 relativePose, json& j )
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
int main ( int argc, char **argv )
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
    ReadImage ( jpg_filenameL.c_str (), &imageL );
    ReadImage ( jpg_filenameR.c_str (), &imageR );

    //--
    // Detect regions thanks to an image_describer
    //--
    using namespace openMVG::features;
    std::unique_ptr<Image_describer> image_describer ( new SIFT_Anatomy_Image_describer );
    std::map<IndexT, std::unique_ptr<features::Regions> > regions_perImage;
    image_describer->Describe ( imageL, regions_perImage[0] );
    image_describer->Describe ( imageR, regions_perImage[1] );

    const SIFT_Regions* regionsL = dynamic_cast<SIFT_Regions*>(regions_perImage.at ( 0 ).get ());
    const SIFT_Regions* regionsR = dynamic_cast<SIFT_Regions*>(regions_perImage.at ( 1 ).get ());

    const PointFeatures
        featsL = regions_perImage.at ( 0 )->GetRegionsPositions (),
        featsR = regions_perImage.at ( 1 )->GetRegionsPositions ();

    // Show both images side by side
    {
        Image<unsigned char> concat;
        ConcatH ( imageL, imageR, concat );
        string out_filename = "01_concat.jpg";
        WriteImage ( out_filename.c_str (), concat );
    }

    //- Draw features on the two image (side by side)
    {
        Features2SVG
        (
            jpg_filenameL,
            { imageL.Width (), imageL.Height () },
            regionsL->Features (),
            jpg_filenameR,
            { imageR.Width (), imageR.Height () },
            regionsR->Features (),
            "02_features.svg"
        );
    }

    std::vector<IndMatch> vec_PutativeMatches;
    //-- Perform matching -> find Nearest neighbor, filtered with Distance ratio
    {
        // Find corresponding points
        matching::DistanceRatioMatch (
            0.8, matching::BRUTE_FORCE_L2,
            *regions_perImage.at ( 0 ).get (),
            *regions_perImage.at ( 1 ).get (),
            vec_PutativeMatches );

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
            { imageL.Width (), imageL.Height () },
            regionsL->GetRegionsPositions (),
            jpg_filenameR,
            { imageR.Width (), imageR.Height () },
            regionsR->GetRegionsPositions (),
            vec_PutativeMatches,
            "03_Matches.svg",
            bVertical
        );
    }

    // Essential geometry filtering of putative matches
    {
        Mat3 K;
        //read K from file
        if ( !readIntrinsic ( camera_K_file, K ) )
        {
            std::cerr << "Cannot read intrinsic parameters." << std::endl;
            return EXIT_FAILURE;
        }

        //A. prepare the corresponding putatives points
        Mat xL ( 2, vec_PutativeMatches.size () );
        Mat xR ( 2, vec_PutativeMatches.size () );
        for ( size_t k = 0; k < vec_PutativeMatches.size (); ++k ) {
            const PointFeature & imaL = featsL[vec_PutativeMatches[k].i_];
            const PointFeature & imaR = featsR[vec_PutativeMatches[k].j_];
            xL.col ( k ) = imaL.coords ().cast<double> ();
            xR.col ( k ) = imaR.coords ().cast<double> ();
        }

        //B. Compute the relative pose thanks to a essential matrix estimation
        std::pair<size_t, size_t> size_imaL ( imageL.Width (), imageL.Height () );
        std::pair<size_t, size_t> size_imaR ( imageR.Width (), imageR.Height () );
        
        
        vector<cv::Point2f> pts_2d_1, pts_2d_2;
        

        cv::Mat K_OCV;
        cv::eigen2cv ( K, K_OCV );
        cout << "K_eigen: " << K << endl;
        cout << "K_OCV: " << K_OCV << endl;


        for ( size_t k = 0; k < xL.cols (); ++k ) {

            cv::Point2d kp1 ( xL.col ( k ).x (), xL.col ( k ).y () );
            cv::Point2d kp2 ( xR.col ( k ).x (), xR.col ( k ).y () );

            pts_2d_1.push_back ( kp1 );
            pts_2d_2.push_back ( kp2 );
        }        

        cv::Mat R_OCV, t_OCV;



        double inliers_ratio = 0.0;
        _calculateRT_CV3_with_ratio ( pts_2d_1, pts_2d_2, K_OCV, R_OCV, t_OCV, inliers_ratio );
        cout << "By opencv:" << endl;
        cout << "R_OCV:\n" << R_OCV << endl;
        cout << "t_OCV:\n" << t_OCV << endl;
        cout << "inliers ration: " << inliers_ratio << endl;
        
        
        // Save R,t to file
        json j;
        j["im1"] = im1;
        j["im2"] = im2;

        std::vector<double> K_vec ( 9 );
        Eigen::Map<Eigen::MatrixXd> ( K_vec.data (), 3, 3 ) = K.transpose ();
        j["K"] = K_vec;


        j["R"] = getVector ( R_OCV );
        j["t"] = getVector ( t_OCV );

        
        
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
