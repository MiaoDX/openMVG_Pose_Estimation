#pragma once
#include <array>

#include "openMVG/multiview/solver_essential_kernel.hpp"
#include "openMVG/multiview/triangulation.hpp"
#include "openMVG/numeric/numeric.h"
#include "openMVG/robust_estimation/robust_estimator_ACRansac.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansacKernelAdaptator.hpp"

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

/// OpenCV Includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"


using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::sfm;
using namespace std;

double AFD (Mat xL, Mat xR)
{
    double afd = 0.0;
    if ( xL.cols () != xR.cols () || xL.rows() != xR.rows() || xL.rows()!=2 ) {
        std::cerr << "The matrixs are somewhat wrong" << std::endl;
        return EXIT_FAILURE;
    }
    
    cout << "squaredNorm:" << (xL - xR).squaredNorm () << endl;
    cout << "Norm:" << (xL - xR).norm () << endl;
    afd = (xL - xR).norm ()/xL.cols();
    
    return afd;
}

double AFD2 ( Mat xL, Mat xR )
{
    double afd = 0.0;
    if ( xL.cols () != xR.cols () || xL.rows () != xR.rows () || xL.rows () != 2 ) {
        std::cerr << "The matrixs are somewhat wrong" << std::endl;
        return EXIT_FAILURE;
    }

    Mat delta = xL - xR;

    double squaredNorm = 0.0;
    for ( size_t i = 0; i < delta.cols(); i++ )
    {   
        double x = delta.col ( i ).x ();
        double y = delta.col ( i ).y ();
        squaredNorm += (x*x + y*y);
    }


    cout << "squaredNorm:" << squaredNorm << endl;
    cout << "Norm:" << sqrt( squaredNorm) << endl;
    afd = sqrt ( squaredNorm ) / xL.cols ();

    return afd;

    return afd;
}