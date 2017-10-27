#include "robustRelativePoseHelper.h"
#include "openMVG/geometry/rigid_transformation3D_srt.hpp"

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::sfm;
using namespace std;

int main()
{
	// Simulate two point set, apply a known transformation and estimate it back:
	const int nbPoints = 10;
	const Mat x1 = Mat::Random(3,nbPoints);
	Mat x2 = x1;

	const double scale = 2.0;
	//const Mat3 rot = (Eigen::AngleAxis<double>(.2, Vec3::UnitX())
	//  * Eigen::AngleAxis<double>(.3, Vec3::UnitY())
	//  * Eigen::AngleAxis<double>(.6, Vec3::UnitZ())).toRotationMatrix();

    /*const Mat3 rot = (Eigen::AngleAxis<double> ( .2, Vec3::UnitX () )
        * Eigen::AngleAxis<double> ( .3, Vec3::UnitY () )
        * Eigen::AngleAxis<double> ( .6, Vec3::UnitZ () )).toRotationMatrix ();*/
    
    double roll = 5, yaw = 0, pitch = 5;
    Eigen::AngleAxisd rollAngle ( roll, Eigen::Vector3d::UnitZ () );
    Eigen::AngleAxisd yawAngle ( yaw, Eigen::Vector3d::UnitY () );
    Eigen::AngleAxisd pitchAngle ( pitch, Eigen::Vector3d::UnitX () );

    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;

    Eigen::Matrix3d rotationMatrix = q.matrix ();

    cout << "ROLL, YAW, PITCH:" << rotationMatrix << endl;

    Eigen::Quaternion<double> q2 = pitchAngle * yawAngle * rollAngle;

    Eigen::Matrix3d rotationMatrix2 = q2.matrix ();

    cout << "PITCH, YAW, ROLL:" << rotationMatrix2 << endl;

    const Mat3 rot = q2.toRotationMatrix ();

    const Vec3 t(0.5,-0.3,.38);

	for (int i=0; i < nbPoints; ++i)
	{
	  const Vec3 pt = x1.col(i);
	  x2.col(i) = (scale * rot * pt + t);
	}

	// Compute the Similarity transform
	double Sc;
	Mat3 Rc;
	Vec3 tc;
	FindRTS(x1, x2, &Sc, &tc, &Rc);
	// Optional non linear refinement of the found parameters
	Refine_RTS(x1,x2,&Sc,&tc,&Rc);

	std::cout << "\n"
	  << "Scale " << Sc << "\n"
	  << "Rot \n" << Rc << "\n"
	  << "t " << tc.transpose();

	std::cout << "\nGT\n"
	  << "Scale " << scale << "\n"
	  << "Rot \n" << rot << "\n"
	  << "t " << t.transpose();
	}