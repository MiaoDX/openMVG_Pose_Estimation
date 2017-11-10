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
    string output_json_file = "output.json";

    CmdLine cmd;
    cmd.add ( make_option ( 'a', im1, "image1" ) );
    cmd.add ( make_option ( 'b', im2, "image2" ) );
    cmd.add ( make_option ( 'o', output_json_file, "output_json file" ) );



    try {
        if ( argc == 1 ) throw std::string ( "Invalid command line parameter." );
        cmd.process ( argc, argv );
    }
    catch ( const std::string& s ) {
        std::cerr << "Usage: " << argv[0] << ' '
            << "[-a|--image1 - the file name of image1, absolute path, eg. H:/dataset/1.jpg]\n"
            << "[-b|--image2 - the name of image2]\n"
            << "[-o|--output_json_file - json file for the R,t]\n"
            << std::endl;

        std::cerr << s << std::endl;
        return EXIT_FAILURE;
    }


  Image<RGBColor> image;

  const string jpg_filenameL = im1;
  const string jpg_filenameR = im2;

  std::vector<IndMatch> vec_PutativeMatches;
  Mat xL, xR, xL_Hfiltering, xR_Hfiltering;
  
  get_matches ( jpg_filenameL, jpg_filenameR, vec_PutativeMatches, xL, xR, xL_Hfiltering, xR_Hfiltering );

}
