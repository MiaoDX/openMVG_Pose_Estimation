/// OpenCV Includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"


#include <vector>
#include <iostream>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
static boost::variate_generator<boost::mt19937, boost::normal_distribution<> >
nd_generator ( boost::mt19937 ( 0/*time(0)*/ ), boost::normal_distribution<> () ); //normal distribution generator


using namespace std;

template< typename T >
T lateral_noise_kinect ( T theta_, T z_, T f_ )
{
    //Nguyen, C.V., Izadi, S., &Lovell, D. (2012).Modeling kinect sensor noise for improved 3D reconstruction and tracking.In 3DIM / 3DPVT (pp. 524?30).http://doi.org/10.1109/3DIMPVT.2012.84
    T sigma_l;
    sigma_l = T ( .8 ) + T ( .035 )*theta_ / (T ( M_PI / 2. ) - theta_);
    sigma_l = sigma_l * z_ / f_;

    //cout << "in lateral_noise_kinect:" << endl;

    return sigma_l;
}

template< typename T >
T axial_noise_kinect ( T theta_, T z_ ) {
    T sigma_a;
    if ( fabs ( theta_ ) <= T ( M_PI / 3. ) )
        sigma_a = T ( .0012 ) + T ( .0019 )*(z_ - 0.4)*(z_ - 0.4);
    else
        sigma_a = T ( .0012 ) + T ( .0019 )*(z_ - 0.4)*(z_ - 0.4) + T ( .0001 ) * theta_* theta_ / sqrt ( z_ ) / (M_PI / 2 - theta_) / (M_PI / 2 - theta_);
    return sigma_a;
}




template< typename T >
void simulate_kinect_noise_on_3d ( vector<cv::Vec3b>& nl_c_gt, vector<cv::Point3f>& pt_c_gt, T f_ )
{
    /*
    for (int i = 0; i < number_; i++) {
		T theta = acos(nl_c_gt.col(i).dot(Matrix<T, 3, 1>(0, 0, -1)));
		T z = pt_c_gt.col(i)(2);
		T sigma_l = lateral_noise_kinect<T>(theta, z, f_);
		T sigma_a = axial_noise_kinect<T>(theta, z);
		Matrix<T, 3, 1> random_variable(sigma_l*nd_generator(), sigma_l*nd_generator(), sigma_a*nd_generator());
		p_pt_c_->col(i) = pt_c_gt.col(i) + random_variable;
		all_weights(i, 1) = short(sigma_min / sigma_a * numeric_limits<short>::max());
	}
     */


    int number_ = pt_c_gt.size ();

    cout << "-------- Before noise --------" << endl;
    for(int i =0;i < 10; i++ )
    {   
        cout << "i:" << i << ", nl_c_gt:" << nl_c_gt[i] << endl;
        cout << "i:" << i << ", pt_c_gt:" << pt_c_gt[i] << endl;
    }
    

    for ( int i = 0; i < number_; i++ ) {
        cout << "i:" << i << endl;
        
        
        //T theta = acos ( nl_c_gt.col ( i ).dot ( Matrix<T, 3, 1> ( 0, 0, -1 ) ) );
        Eigen::Matrix<T, 1, 3> nl_c_gt_i ( nl_c_gt[i][0], nl_c_gt[i][1], nl_c_gt[i][2] );
        nl_c_gt_i.normalize ();

        T theta = acos (( nl_c_gt_i ).dot ( Eigen::Matrix<T, 3, 1> ( 0, 0, -1 ) ));


        cout << "theta:" << theta << endl;

        //T z = pt_c_gt.col ( i )(2);
        T z = pt_c_gt[i].z;

        T sigma_l = lateral_noise_kinect<T> ( theta, z, f_ );
        T sigma_a = axial_noise_kinect<T> ( theta, z );


        cout << "sigma_l:" << sigma_l << endl << "sigma_a:" << sigma_a << endl;

        //Matrix<T, 3, 1> random_variable ( sigma_l*nd_generator (), sigma_l*nd_generator (), sigma_a*nd_generator () );
        cv::Point3f delta_p ( sigma_l*nd_generator (), sigma_l*nd_generator (), sigma_a*nd_generator () );


        cout << "delta_p:\n" << delta_p << endl;

        //p_pt_c_->col ( i ) = pt_c_gt.col ( i ) + random_variable;
        pt_c_gt[i] += delta_p;
    }

    cout << "-------- After noise --------" << endl;
    for ( int i = 0; i < 10; i++ )
    {
        cout << "i:" << i << ", pt_c_gt:" << pt_c_gt[i] << endl;
    }
}