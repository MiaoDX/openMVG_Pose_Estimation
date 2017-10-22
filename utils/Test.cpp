#include "AFD.h"

using namespace std;

int main ()
{
    Mat m ( 2, 4 );
    Mat n ( 2, 4 );

    m << 1, 23, 6, 9,
        3, 11, 7, 2;

    n << 0, 23, 5, 8,
        2, 11, 6, 2;


    for ( size_t i = 0; i < m.cols(); i++ )
    {
        double x1 = m.col ( i ).x ();
        double y1 = m.col ( i ).y ();

        double x2 = n.col ( i ).x ();
        double y2 = n.col ( i ).y ();
        cout << "x1:" << x1 << ",y1:" << y1 << ",x2:" << x2 << ",y2:" << y2 << endl;
    }

    double afd = AFD ( m, n );

    cout << "AFD:" << afd << endl;


    double afd2 = AFD2 ( m, n );

    cout << "AFD2:" << afd2 << endl;

    cout << "haha" << endl;
    system ( "pause" );
}