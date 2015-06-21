#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

//#include "cv.h"
//#include "highgui.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "filter.h"
#include "hough_transform.h"
#include "image_wrapper.h"


#include <algorithm>
#include <limits>
#include <string>
#include <math.h>

using namespace std;



/*
  follow object recognition pipeline from Prof. Leibe's tutorial (but simpler :p)
  0. given a good training image containing the object to be tracked
     (or during initialization, let user choose region of interest i.e. sub-image)
  1. use harris detector to find corners
  2. extract local region descriptors (sift, maglap etc.)
  3. find best matching correspondence set (for each feature find its nearest neighbor)
  4. geometric verification (using Homography estimation and Ransac for noise removal (
     using symmetric transfer error)).
     (affine transformation is ok, fundamental matrix is not robust in this case).
     count #inliers as hypothesis score
*/
class Object_Detection
{
public:
    Object_Detection();

    std::vector<Hough_Transform::line> harris(IplImage* img, float sigma, float threshold);



    void descriptor_maglap(IplImage* img, const std::vector<Hough_Transform::line> & corners
                           , int size, float sigma, int bins, vector< vector<float> > & desp);

    void doMatch(IplImage* model, IplImage* data, float sigma, float threshold, int k);

    void doMatch_with_ransac(IplImage* model, IplImage* data, float sigma, float threshold, int k);

    CvPoint doMatch_with_ransac(const std::vector<Hough_Transform::line> & harrisModel,
                             const vector< vector<float> > & despModel,
                             IplImage *data, float sigma, float threshold, int k);

    IplImage* subImage(IplImage* img, int minx, int miny, int maxx, int maxy);

    std::vector<CvPoint> do_local_match(const vector< vector<float> > & despModel,
                           IplImage *data, float sigma, float threshold);

    struct matching
    {
        int from;
        int to;
        float distance;
    };

private:

    Filter filter;
    Hough_Transform hough;

    /*
     * return vector h of 2D histogram for gradient magnitude and laplacian
     * this is a simple descriptor
     * h: define a 2D histogram  with "bins^2" number of entries
    */
    void magLapHist(IplImage* img, float sigma, int bins, vector<float> & h);

    /*
      compute chi2 distance from 2 histograms
    */
    float hist_dist_chi2(const vector<float> & h1, const vector<float> & h2);



    void find_nn_chi2(const vector< vector<float> > & desp1
                      , const vector< vector<float> > & desp2, std::vector<matching> & matchings);

    string convertInt(int number);

    void showCorners(IplImage *img, const std::vector<Hough_Transform::line> & modelCorners
                     , IplImage * img2, const std::vector<Hough_Transform::line> & dataCorners
                     , const std::vector<matching> & matchings, int k);


    IplImage * showCorners(IplImage* img, const std::vector<Hough_Transform::line> & corners
                           , const std::vector<matching> & matchings);


    /*
        geometric verification for difficult situations.
    */
    void homography_estimation(const std::vector<Hough_Transform::line> & modelCorners
                               , const std::vector<Hough_Transform::line> & dataCorners
                               , const std::vector<matching> & matchings, int k
                               , float homo[][3]
        );

    /*
      to remove noise
    */
    void ransac(const std::vector<Hough_Transform::line> & modelCorners
                , const std::vector<Hough_Transform::line> & dataCorners
                , const std::vector<matching> & matchings, int k
                , float homo[][3]
                , std::vector<matching> & inliers, int iterations);

    /**
     * @brief transform model corners to data image. Be careful about float/in coordinates
     *
     * @param modelCorners
     * @param dataCorners
     * @param k
     * @param homo[][]
     */
    void transform(const std::vector<Hough_Transform::line> & modelCorners
                   , std::vector<Hough_Transform::line> & dataCorners
                   , int k, float homo[][3]
                   , const std::vector<matching> & matchings
        );


    void get_inliers(const std::vector<Hough_Transform::line> & modelCorners
                     , const std::vector<Hough_Transform::line> & dataCorners
                     , const std::vector<matching> & matchings, int k
                     , float homo[][3]
                     , std::vector<matching> & inliers, float squared_dist_threshold
            );


    CvPoint draw_object_bbox(std::vector<Hough_Transform::line> harrisData,
                          std::vector<matching> inliers, IplImage *data);

    int sign(float value);
    bool is_good_pose(const std::vector<Hough_Transform::line> & modelCorners
                     , const std::vector<Hough_Transform::line> & dataCorners
                     , const std::vector<matching> & matchings);


};




class CmpDist
{
public:
  CmpDist() {}

  bool operator()(const Object_Detection::matching & m1, const Object_Detection::matching & m2) const
  {

      return m1.distance < m2.distance;

  }

};



#endif // OBJECT_DETECTION_H
