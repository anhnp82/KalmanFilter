#ifndef FILTER_H
#define FILTER_H

//#include "highgui.h"
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
  only support float images since some operations can result in float images
*/

class Filter
{

private:

    //mark the visited pixels in the follow edge algorithm
    struct FollowResult
    {
        IplImage* visited;
        IplImage* edgeImage;
    } ;

public:
    Filter();

    void gauss(float g[], int length, float sigma);

    void gaussDx(float g[], int length, float sigma);

    /*convolve x with y*/
    float dot(float x[], float y[], int length);

    IplImage* gaussianFilter(IplImage* img, float sigma, bool floatImg = false);

    IplImage* myCanny(IplImage* img, float sigma, float lower, float upper);

    IplImage* convolution(IplImage* img_b, float xKernel[], float yKernel[], int w, bool floatImg = false);

    IplImage * genericFilter(IplImage *img, float xKernel[], float yKernel[], int w, bool floatImg = false);

    IplImage* imageXDerivative(IplImage *img, float sigma, bool floatImg = false);

    IplImage* imageYDerivative(IplImage *img, float sigma, bool floatImg = false);

    IplImage* laplace(IplImage *img, float sigma);

    IplImage* nonMaxSupCanny(IplImage* imgMag,IplImage* imgDir);

    IplImage* gradientMagnitudeCanny(IplImage* img, float sigma, float & maxGrad);

    IplImage* gradMag(IplImage* img, float sigma);

    IplImage* thresholding(IplImage* gradMax, float threshold);

    FollowResult followEdge(int i, int j, FollowResult edgeResult);

    void showImage(IplImage* img);

};

#endif // FILTER_H
