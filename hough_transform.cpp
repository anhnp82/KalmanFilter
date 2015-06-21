#include "hough_transform.h"
#include "image_wrapper.h"
//#include <math.h>
#define _USE_MATH_DEFINES
//#include <cmath>
#include <math.h>
#include "filter.h"
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

Hough_Transform::Hough_Transform()
{

}


IplImage* Hough_Transform::transformToHoughSpace(IplImage* edgeImage, int nBinsRho, int nBinsTheta)
{

    CvSize s = cvSize(nBinsRho, nBinsTheta);
    //Accumulator
    IplImage* hough = cvCreateImage(s, IPL_DEPTH_32F, 1 ); //IPL_DEPTH_32S
    cvZero(hough);

    //Image diagonal
    //float D = sqrt(pow(edgeImage->height, 2) + pow(edgeImage->width, 2));
	float D = sqrt((float)edgeImage->height*edgeImage->height + edgeImage->width*edgeImage->width);

    BwImageFloat edgeWrapper(edgeImage);
    BwImageFloat houghWrapper(hough);

    // Hough-Transformation
    for (int x = 0; x < edgeImage->width; x++ )
    {
        for (int y = 0; y < edgeImage->height; y++)
        {
            if (edgeWrapper[y][x] == 255)
            {
                for (int n = 0; n < nBinsTheta; n++)
                {
                    float theta = -M_PI/2 + M_PI * (n)/(nBinsTheta-1);
                    float rho = x*cos(theta) + y*sin(theta);
                    int m = round((nBinsRho-1)*(rho+D)/(2*D));
                    houghWrapper[m][n] += 1;
                }
            }
        }
    }

    //showImage(hough);
    return hough;

}

IplImage* Hough_Transform::nonMaxSuppress(IplImage *hough)
{

    IplImage* imgResult = cvCreateImage( cvGetSize(hough), IPL_DEPTH_32F, 1 );
    cvZero(imgResult);

    BwImageFloat houghWrap(hough);
    BwImageFloat resultWrap(imgResult);

    for (int y = 1; y < hough->height-1; y++)
    {
        for (int x = 1; x < hough->width-1; x++)
        {

            int offx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
            int offy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

            int val = houghWrap[y][x];

            bool is_max = true;

            for (int i=0; i < 8; i++)
            {
                if (val < houghWrap[y+offy[i]][x+offx[i]])
                    is_max = false;
            }

            if (is_max)
                resultWrap[y][x] = val;

        }
    }
    return imgResult;
}

std::vector<Hough_Transform::line> Hough_Transform::findhoughpeaks(IplImage *hough, float threshold)
{
    IplImage* houghMax = nonMaxSuppress(hough);

    //printf("non max ok \n");
    //showImage(houghMax);

    BwImageFloat houghWrap(houghMax);
    std::vector<line> lines;

    for (int i = 0; i < houghMax->height; i++)
    {
        for (int j = 0; j < houghMax->width; j++)
        {
            if (houghWrap[i][j] >= threshold)
            {
                line houghLine;
                houghLine.rho = i;
                houghLine.theta = j;
                houghLine.peak = houghWrap[i][j];
                lines.push_back(houghLine);
            }
        }
    }

    cvReleaseImage( &houghMax );

    return lines;
}

IplImage* Hough_Transform::drawHoughLines(IplImage* img, IplImage *edgeImage, int peak, int nBins)
{

    //Filter filter;
    //IplImage* edgeImage = filter.myCanny(img, sigma, lower, upper);//0.3, 0.5, 0.8

    int nBinsRho = nBins;
	int nBinsTheta = nBins;
    
    int img_width = edgeImage->width-1;
    //float D = sqrt(pow(edgeImage->height, 2) + pow(edgeImage->width, 2));
	float D = sqrt((float)edgeImage->height*edgeImage->height + edgeImage->width*edgeImage->width);

    IplImage* houghSpace = transformToHoughSpace(edgeImage, nBinsRho, nBinsTheta);

    printf("transformToHoughSpace ok \n");
    //showImage(houghSpace);

    std::vector<line> houghLines = findhoughpeaks(houghSpace, peak);//180


    printf("find peaks ok \n");

    IplImage* result = cvCloneImage( img );
    //cvCopyImage( img, result );
	//cvCopy( img, result );

    for (int i = 0; i < houghLines.size(); i++)
    {
        //cvLine(img, cvPoint(100,100), cvPoint(200,200), cvScalar(0,255,0), 1);
        int n = houghLines[i].theta;
        int m = houghLines[i].rho;
        float theta = -M_PI/2 + M_PI * (n)/(nBinsTheta-1);

        //if (180*theta/M_PI < 0 ) continue;

        float rho = (m)*2*D/(nBinsRho-1)-D;

        CvPoint X = cvPoint((int)0, (int)((rho - cos(theta)) / sin(theta)));
        CvPoint Y = cvPoint((int)img_width, (int)((rho - img_width * cos(theta)) / sin(theta)));
        cvLine(result, X, Y, cvScalar(255), 1);
    }

    cvReleaseImage( &houghSpace );
    cvReleaseImage( &edgeImage );

    return result;
}


void Hough_Transform::showImage(IplImage *img)
{
    cvNamedWindow( "test", CV_WINDOW_NORMAL );
    cvShowImage( "test", img );
    cvWaitKey(0);
    cvReleaseImage( &img );
    cvDestroyWindow( "test" );
}
