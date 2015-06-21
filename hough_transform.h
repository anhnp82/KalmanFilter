#ifndef HOUGH_TRANSFORM_H
#define HOUGH_TRANSFORM_H

//#include "cv.h"
//#include "highgui.h"

#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

/*
  x-intercept = p/cos(theta)
  y-intercept = p/sin(theta)
  the point(x, y) in the line segment has the following constraint:
  y/y-intercept = (x-intercept - x)/x-intercept
  this leads to the following constraint:
  x*cos(theta) + y*sin(theta) = p
  where theta in [-90, 90], p = image diagonal length
*/

/*
  There are many extensions, we only implement the basic version.
  we also binalize the image by using a simplified version of canny edge detection.
  This will reduce the number of pixels to be processed in hough transform
  i.e. only those on edges are processed.
*/

class Hough_Transform
{

//#define nBinsRho 300
//#define nBinsTheta 300

public:
    Hough_Transform();
    IplImage* drawHoughLines(IplImage* img, IplImage* edgeImage, int peak, int nBins);

    struct line
    {
        int rho;
        int theta;
        float peak;
    };

    std::vector<line> findhoughpeaks(IplImage* hough, float threshold);

private:



    IplImage * transformToHoughSpace(IplImage* edgeImage, int nBinsRho, int nBinsTheta);
    IplImage * nonMaxSuppress(IplImage* hough);
    void showImage(IplImage* img);

};

#endif // HOUGH_TRANSFORM_H
