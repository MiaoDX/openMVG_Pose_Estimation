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

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::robust;
using namespace std;


cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K )
{
    return cv::Point2d
    (
        (p.x - K.at<double> ( 0, 2 )) / K.at<double> ( 0, 0 ),
        (p.y - K.at<double> ( 1, 2 )) / K.at<double> ( 1, 1 )
    );
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



void get_matches(string jpg_filenameL, string jpg_filenameR, std::vector<IndMatch>& vec_PutativeMatches,
    Mat& xL, Mat& xR, Mat& xL_Hfiltering, Mat& xR_Hfiltering )
{
    Image<unsigned char> imageL, imageR;
    ReadImage ( jpg_filenameL.c_str (), &imageL );
    ReadImage ( jpg_filenameR.c_str (), &imageR );

    //--
    // Detect regions thanks to an image_describer
    //--
    using namespace openMVG::features;
    std::unique_ptr<Image_describer> image_describer
    ( new SIFT_Anatomy_Image_describer ( SIFT_Anatomy_Image_describer::Params ( -1 ) ) );
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


    // Homography geometry filtering of putative matches
    {
        xL.resize ( 2, vec_PutativeMatches.size () );
        xR.resize ( 2, vec_PutativeMatches.size () );

        for ( size_t k = 0; k < vec_PutativeMatches.size (); ++k ) {
            const PointFeature & imaL = featsL[vec_PutativeMatches[k].i_];
            const PointFeature & imaR = featsR[vec_PutativeMatches[k].j_];
            xL.col ( k ) = imaL.coords ().cast<double> ();
            xR.col ( k ) = imaR.coords ().cast<double> ();
        }

        cout << "xL:" << xL.cols () << endl;
        //for ( size_t i = 0; i < xL.cols (); ++i ) {
        //    cout << xL.col ( i ) << endl << endl;
        //}

        //-- Homography robust estimation
        std::vector<uint32_t> vec_inliers;
        using KernelType =
            ACKernelAdaptor<
            openMVG::homography::kernel::FourPointSolver,
            openMVG::homography::kernel::AsymmetricError,
            UnnormalizerI,
            Mat3>;

        KernelType kernel (
            xL, imageL.Width (), imageL.Height (),
            xR, imageR.Width (), imageR.Height (),
            false ); // configure as point to point error model.

        Mat3 H;
        const std::pair<double, double> ACRansacOut = ACRANSAC ( kernel, vec_inliers, 1024, &H,
            std::numeric_limits<double>::infinity (),
            true );
        const double & thresholdH = ACRansacOut.first;

        // Check the homography support some point to be considered as valid
        if ( vec_inliers.size () > KernelType::MINIMUM_SAMPLES *2.5 ) {

            std::cout << "\nFound a homography under the confidence threshold of: "
                << thresholdH << " pixels\n\twith: " << vec_inliers.size () << " inliers"
                << " from: " << vec_PutativeMatches.size ()
                << " putatives correspondences"
                << std::endl;

            //Show homography validated point and compute residuals
            const bool bVertical = true;
            InlierMatches2SVG
            (
                jpg_filenameL,
                { imageL.Width (), imageL.Height () },
                regionsL->GetRegionsPositions (),
                jpg_filenameR,
                { imageR.Width (), imageR.Height () },
                regionsR->GetRegionsPositions (),
                vec_PutativeMatches,
                vec_inliers,
                "04_ACRansacHomography.svg",
                bVertical
            );


            xL_Hfiltering.resize ( 2, vec_inliers.size () );
            xR_Hfiltering.resize ( 2, vec_inliers.size () );


            std::vector<double> vec_residuals ( vec_inliers.size (), 0.0 );
            for ( size_t i = 0; i < vec_inliers.size (); ++i ) {
                const SIOPointFeature & LL = regionsL->Features ()[vec_PutativeMatches[vec_inliers[i]].i_];
                const SIOPointFeature & RR = regionsR->Features ()[vec_PutativeMatches[vec_inliers[i]].j_];
                // residual computation
                vec_residuals[i] = std::sqrt ( KernelType::ErrorT::Error ( H,
                    LL.coords ().cast<double> (),
                    RR.coords ().cast<double> () ) );
                xL_Hfiltering.col ( i ) = LL.coords ().cast<double> ();
                xR_Hfiltering.col ( i ) = RR.coords ().cast<double> ();
            }

            cout << "xL_Hfiltering:" << xL_Hfiltering.cols () << endl;
            //for ( size_t i = 0; i < xL_Hfiltering.cols (); ++i ) {
            //    cout << xL_Hfiltering.col ( i ) << endl << endl;
            //}


            // Display some statistics of reprojection errors
            float dMin, dMax, dMean, dMedian;
            minMaxMeanMedian<float> ( vec_residuals.begin (), vec_residuals.end (),
                dMin, dMax, dMean, dMedian );

            std::cout << std::endl
                << "Homography matrix estimation, residuals statistics:" << "\n"
                << "\t-- H matrix:\t" << H << std::endl
                << "\t-- Residual min:\t" << dMin << std::endl
                << "\t-- Residual median:\t" << dMedian << std::endl
                << "\t-- Residual max:\t " << dMax << std::endl
                << "\t-- Residual mean:\t " << dMean << std::endl;

     
            Image<RGBColor> image;
            //---------------------------------------
            // Warp the images to fit the reference view
            //---------------------------------------
            // reread right image that will be warped to fit left image
            ReadImage ( jpg_filenameR.c_str (), &image );
            WriteImage ( "query.png", image );

            // Create and fill the output image
            Image<RGBColor> imaOut ( imageL.Width (), imageL.Height () );
            image::Warp ( image, H, imaOut );
            const std::string imageNameOut = "query_warped.png";
            WriteImage ( imageNameOut.c_str (), imaOut );
        }
        else {
            std::cout << "ACRANSAC was unable to estimate a rigid homography"
                << std::endl;
        }
    }


}