#include "filter.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "image_wrapper.h"

Filter::Filter()
{
}

void Filter::gauss(float g[], int length, float sigma)
{
    int w = (length - 1)/2;
    for (int i = 0; i < length; i++)
    {
        int x = i - w;
        g[i] = (1/(sqrt(2*M_PI)*sigma)) * exp(-((x*x)/(2*sigma*sigma)));
    }
}

void Filter::gaussDx(float g[], int length, float sigma)
{
    int w = (length - 1)/2;
    for (int i = 0; i < length; i++)
    {
        int x = i - w;
        //g[i] = (1/(sqrt(2*M_PI)*sigma)) * exp(-((x*x)/(2*sigma*sigma)));
        g[i] = -(x*exp(-x*x/(2*sigma*sigma)))/(sqrt(2*M_PI)*pow(sigma, 3));
    }
}

float Filter::dot(float x[], float y[], int length)
{
    float sum = 0;
    for (int i = 0; i < length; i++)
        sum += x[i]*y[i];
    return sum;
}

IplImage* Filter::convolution(IplImage* img, float xKernel[], float yKernel[], int w, bool floatImg)
{

    int length = 2*w + 1;
	float* y = new float[length];

    IplImage* out = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    IplImage* tmp = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(tmp);
    cvZero(out);

    BwImage wrapImg_b(img);
    BwImageFloat wrapImg_bFloat(img);
    BwImageFloat wrapOut(out);
    BwImageFloat wrapTmp(tmp);

    //double max = -100000000;

    //filter in x direction
    for (int i = 0; i < img->height; i++)//row
    {
        for (int j = w; j < img->width - w; j++)//column
        {
            //float y[length];
            for (int k = -w; k <= w; k++)
            {
                if (!floatImg) y[k+w] = wrapImg_b[i][j+k];
                else
                {
                    y[k+w] = wrapImg_bFloat[i][j+k];
                    //if (y[k+w] > max) max = y[k+w];
                    //printf("float intensity \n", y[k+w]);
                }
            }
            wrapTmp[i][j] = dot(y, xKernel, length);
        }
    }


    //filter in y direction
    for (int i = w; i < img->height - w; i++)//row
    {
        for (int j = 0; j < img->width; j++)//column
        {
            //float y[length];
            for (int k = -w; k <= w; k++)
            {
                y[k+w] = wrapTmp[i+k][j];
            }
            wrapOut[i][j] = dot(y, yKernel, length);
        }
    }

    cvReleaseImage( &tmp );

    //printf("max intensity %f \n", max);

	delete[] y;

    return out;
}

IplImage * Filter::genericFilter(IplImage *img, float xKernel[], float yKernel[], int w, bool floatImage)
{

    // Add convolution boarders
    CvPoint offset = cvPoint(w, w);
    IplImage* img_b = cvCreateImage( cvSize(img->width+2*w,img->height+2*w), img->depth, 1 );
    cvZero(img_b);

    cvCopyMakeBorder(img, img_b, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));
	//copyMakeBorder(img, img_b, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));

    //if (floatImage) cvSaveImage("test.png", img_b, 0);

    //problem in float convolution
    IplImage* out = convolution(img_b, xKernel, yKernel, w, floatImage);

    //if (floatImage) cvSaveImage("test.png", out, 0);

    //clip boundary
    cvSetImageROI(out, cvRect(w, w, out->width - 2*w, out->height - 2*w));
    /* create destination image
       Note that cvGetSize will return the width and the height of ROI */
    IplImage* result = cvCreateImage( cvGetSize(out), IPL_DEPTH_32F, 1 );
    cvZero(result);
    /* copy subimage */
    cvCopy(out, result, NULL);

    /* always reset the Region of Interest */
    cvResetImageROI(out);

    cvReleaseImage( &img_b );
    cvReleaseImage( &out );

    return result;

}

IplImage* Filter::gaussianFilter(IplImage *img, float sigma, bool floatImg)
{
    //generate 1D Gaussian
    int w = ceil(3*sigma);
    int length = 2*w + 1;
    //float g[length];
	float * g = new float[length];

    gauss(g, length, sigma);
    //return genericFilter(img, g, g, w, floatImg);

	IplImage * pFiltered = genericFilter(img, g, g, w, floatImg);

	delete[] g;
	return pFiltered;
}


IplImage * Filter::imageXDerivative(IplImage *img, float sigma, bool floatImg)
{
    //generate 1D Gaussian
    int w = ceil(3*sigma);
    int length = 2*w + 1;
    //float g[length], gx[length];
	float * g = new float[length];
	float * gx = new float[length];

    gauss(g, length, sigma);
    gaussDx(gx, length, sigma);

//    IplImage *imgDx = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
//    cvZero(imgDx);

    //return genericFilter(img, gx, g, w, floatImg);
	IplImage * pFiltered = genericFilter(img, gx, g, w, floatImg);

	delete[] g;
	delete[] gx;

	return pFiltered;
}


IplImage * Filter::imageYDerivative(IplImage *img, float sigma, bool floatImg)
{
    //generate 1D Gaussian
    int w = ceil(3*sigma);
    int length = 2*w + 1;
    //float g[length], gx[length];
	float * g = new float[length];
	float * gx = new float[length];

    gauss(g, length, sigma);
    gaussDx(gx, length, sigma);

//    IplImage *imgDy = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
//    cvZero(imgDy);

    //return genericFilter(img, g, gx, w, floatImg);
	IplImage * pFiltered = genericFilter(img, g, gx, w, floatImg);

	delete[] g;
	delete[] gx;
	return pFiltered;
}

IplImage* Filter::gradientMagnitudeCanny(IplImage* img, float sigma, float & maxGrad)
{
    IplImage* imgDx, *imgDy;

    imgDx = imageXDerivative(img, sigma);
    imgDy = imageYDerivative(img, sigma);

    IplImage* gradMag = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    IplImage* gradDir = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(gradMag);
    cvZero(gradDir);

    BwImageFloat wrapDx(imgDx);
    BwImageFloat wrapDy(imgDy);
    BwImageFloat wrapGradMag(gradMag);
    BwImageFloat wrapGradDir(gradDir);

    //maximum gradient magnitude to be used for scaling high and low thresholds
    maxGrad = 0;

    for (int i = 0; i < gradMag->height; i++)
    {
        for (int j = 0; j < gradMag->width; j++)
        {
            //gradient magnitude
            wrapGradMag[i][j] = sqrt((wrapDx[i][j]*wrapDx[i][j]) + (wrapDy[i][j]*wrapDy[i][j]));
            //gradient orientation range [-Pi, Pi]
            wrapGradDir[i][j] = atan2(wrapDy[i][j],wrapDx[i][j]);
            //compute max gradient magnitude
            if (maxGrad < wrapGradMag[i][j]) maxGrad = wrapGradMag[i][j];
        }
    }

    //cvSaveImage("gradMag.png", gradMag, 0);

    //suppress non-maximum thick edges to get single responses
    IplImage* gradMax = nonMaxSupCanny(gradMag, gradDir);

    cvReleaseImage( &imgDx );
    cvReleaseImage( &imgDy );
    cvReleaseImage( &gradMag );
    cvReleaseImage( &gradDir );

    return gradMax;
}




IplImage* Filter::gradMag(IplImage* img, float sigma)
{
    IplImage* imgDx, *imgDy;

    imgDx = imageXDerivative(img, sigma);
    imgDy = imageYDerivative(img, sigma);

    IplImage* gradMag = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(gradMag);

    BwImageFloat wrapDx(imgDx);
    BwImageFloat wrapDy(imgDy);
    BwImageFloat wrapGradMag(gradMag);

    for (int i = 0; i < gradMag->height; i++)
    {
        for (int j = 0; j < gradMag->width; j++)
        {
            //gradient magnitude
            wrapGradMag[i][j] = sqrt((wrapDx[i][j]*wrapDx[i][j]) + (wrapDy[i][j]*wrapDy[i][j]));
        }
    }

    cvReleaseImage( &imgDx );
    cvReleaseImage( &imgDy );

    return gradMag;
}




IplImage * Filter::laplace(IplImage *img, float sigma)
{
    IplImage* imgDx, *imgDy, *imgDxx, *imgDyy;

    imgDx = imageXDerivative(img, sigma);
    imgDy = imageYDerivative(img, sigma);
    imgDxx = imageXDerivative(imgDx, sigma, true);
    imgDyy = imageYDerivative(imgDy, sigma, true);

    IplImage* laplacian = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(laplacian);


    BwImageFloat wrapDxx(imgDxx);
    BwImageFloat wrapDyy(imgDyy);
    BwImageFloat wrapLap(laplacian);


    for (int i = 0; i < laplacian->height; i++)
    {
        for (int j = 0; j < laplacian->width; j++)
        {

            wrapLap[i][j] = wrapDxx[i][j] + wrapDyy[i][j];

        }
    }


    cvReleaseImage( &imgDx );
    cvReleaseImage( &imgDy );
    cvReleaseImage( &imgDxx );
    cvReleaseImage( &imgDyy );

    return laplacian;
}



IplImage* Filter::myCanny(IplImage* img, float sigma, float lower, float upper)
{
    //compute single-response (max suppression) gradient magnitude
    float maxGrad;
    IplImage* gradMax = gradientMagnitudeCanny(img, sigma, maxGrad);

    //return gradMax;

    float lowThreshold = lower*maxGrad;
    float highThreshold = upper*maxGrad;

    FollowResult edgeResult;

    //mark the visited pixels in the follow edge algorithm
    edgeResult.visited = cvCreateImage( cvGetSize(gradMax), IPL_DEPTH_32F, 1 );
    //mark the starting pixels in the follow edge algorithm
    IplImage* high = cvCreateImage( cvGetSize(gradMax), IPL_DEPTH_32F, 1 );
    //the final edge image
    edgeResult.edgeImage = cvCreateImage( cvGetSize(gradMax), IPL_DEPTH_32F, 1 );

    cvZero(edgeResult.visited);
    cvZero(edgeResult.edgeImage);
    cvZero(high);

    BwImageFloat wrapGradMax(gradMax);
    BwImageFloat wrapVisited(edgeResult.visited);
    BwImageFloat wrapHigh(high);

    for (int i = 0; i < gradMax->height; i++)
    {
        for (int j = 0; j < gradMax->width; j++)
        {
            if (wrapGradMax[i][j] < lowThreshold) wrapVisited[i][j] = 255;
            if (wrapGradMax[i][j] >= highThreshold) wrapHigh[i][j] = 255;
        }
    }

    //return visited;

    // Create boundary around visited flag
    for (int i = 0; i < gradMax->height; i++)
    {
        wrapVisited[i][0] = 255;
        wrapVisited[i][gradMax->width - 1] = 255;
    }

    for (int i = 0; i < gradMax->width; i++)
    {
        wrapVisited[0][i] = 255;
        wrapVisited[gradMax->height - 1][i] = 255;
    }

    //return high;

    // Start at the high pixels and follow the edges
    for (int i=0; i < high->height; i++)
    {

        for(int j=0; j < high->width; j++)
        {
            //printf("high %d \n", wrapHigh[i][j]);
            //if (wrapHigh[i][j] == 255) printf("here \n");

            if((wrapHigh[i][j] == 255) && (wrapVisited[i][j] == 0))
            {
                //printf("begin follow edge at %d %d \n", i, j);
                edgeResult = followEdge(i, j, edgeResult);
            }
        }
    }

    cvReleaseImage( &high );
    cvReleaseImage( &gradMax );

    //showImage(edgeResult.edgeImage);

    return edgeResult.edgeImage;

}

Filter::FollowResult Filter::followEdge(int i, int j, FollowResult edgeResult)
{
    BwImageFloat wrapVisited(edgeResult.visited);
    BwImageFloat wrapEdgeImage(edgeResult.edgeImage);

    wrapVisited[i][j] = 255;
    wrapEdgeImage[i][j] = 255;

    //printf("modified at %d %d \n", i, j);

    int offi[] = {1, 1, 0, -1, -1, -1, 0, 1};
    int offj[] = {0, 1, 1, 1, 0, -1, -1, -1};


    for (int k=0; k < 8; k++)
    {
        int idx_i = i+offi[k];
        int idx_j = j+offj[k];
        if (wrapVisited[idx_i][idx_j] == 0)
            edgeResult = followEdge(idx_i,idx_j, edgeResult);
    }
    return edgeResult;
}

IplImage* Filter::thresholding(IplImage* gradMax, float threshold)
{
    BwImageFloat wrapGradMax(gradMax);

    for (int i = 0; i < gradMax->height; i++)
    {
        for (int j = 0; j < gradMax->width; j++)
        {
            if (wrapGradMax[i][j] < threshold) wrapGradMax[i][j] = 0;
            else wrapGradMax[i][j] = 255;
        }
    }

    return gradMax;
}


IplImage* Filter::nonMaxSupCanny(IplImage* imgMag,IplImage* imgDir)
{

    int h = imgMag->height;
    int w = imgMag->width;

    IplImage* imgMax = cvCreateImage( cvGetSize(imgMag), IPL_DEPTH_32F, 1 );
    cvZero(imgMax);

    BwImageFloat wrapMag(imgMag);
    BwImageFloat wrapDir(imgDir);
    BwImageFloat wrapMax(imgMax);

    //offx[0] and offy[0] correspond to the -PI gradient orientation
    int offx[] = {-1, -1,  0,  1,  1,  1,  0, -1, -1};
    int offy[] = { 0, -1, -1, -1,  0,  1,  1,  1,  0};

    for (int y=1; y < h-1; y++)
    {
        for(int x=1;x < w-1; x++)
        {
        // look up the orientation at that pixel
            float dir = wrapDir[y][x];

        // look up the neighboring pixels in direction of the gradient
        // the output of gradient image is atan2(dy, dx)
        // which has the range [-Pi, Pi]
        // this function converts from this domain to the index of offset arrays [0, 8]
        int idx = floor(((dir+M_PI)/M_PI)*4 + 0.5);

        // suppress all non-maximum points
        if( (wrapMag[y][x] > wrapMag[y+offy[idx]][x+offx[idx]]) &&
                (wrapMag[y][x] > wrapMag[y-offy[idx]][x-offx[idx]]) )
            wrapMax[y][x] = wrapMag[y][x];
      }
    }

    return imgMax;
}

void Filter::showImage(IplImage *img)
{
    cvNamedWindow( "test", CV_WINDOW_NORMAL );
    cvShowImage( "test", img );
    cvWaitKey(0);
    cvReleaseImage( &img );
    cvDestroyWindow( "test" );
}
