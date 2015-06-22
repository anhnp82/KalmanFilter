#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

//#include "cv.h"
//#include "highgui.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "filter.h"
#include "hough_transform.h"
#include "image_wrapper.h"

using namespace cv;
using namespace std;


/*
  0. test: ftp://ftp.pets.rdg.ac.uk/pub/PETS2000/training_images/
           http://ilab.cs.ucsb.edu/tracking_dataset_ijcv/
  1. init: the prediction to some prior knowledge
  2. data association: prediction at time t-1 corresponds to the measurement at the current time t.
     measurement: based on detection (feature matching, corner detection).
  3. correction:
  4. make prediction for the next time step
*/
class Kalman_Filter
{
public:
    Kalman_Filter(float sigma, float threshold, float patch_size);

    Kalman_Filter();

    CvRect get_roi_from_user(IplImage* img);

    void video_extraction();

    void extract_measurement(const char * model_file, const char * data_file
                             , float sigma, float threshold, int k);

    //void start_tracking(const char * file_name, int k = 10, float sigma = 2.0, float threshold = 100000);
    void start_tracking(const char * file_name);

    static CvRect s_box;
    static bool s_bDrawing_box;

    struct point2d
    {
        float x;
        float y;
    };

    ~Kalman_Filter();

private:

    // Define our callback which we will install for
    // mouse events.
    //
    static void my_mouse_callback(
            int event, int x, int y, int flags, void* param
        );

    float sigma, threshold, patch_size;
    int speed;

    std::vector<CvPoint> m_trajectory;

    float dt; //% 25 FPS

    Mat D, M, I;

    Mat sigma_d, sigma_m, sigma_predict;

    Mat x_predict;

    static void draw_box( IplImage* img, CvRect rect );

    IplImage* get_model(IplImage * img1, CvRect & roi);

    void track_object(IplImage* img, vector< vector<float> > despModel);

};



#endif // KALMAN_FILTER_H
