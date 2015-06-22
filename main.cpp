//#include "highgui.h"
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_wrapper.h"
#include "filter.h"
#include "hough_transform.h"
#include "kalman_filter.h"
#include "object_detection.h"

int main( int argc, char** argv )
{

	int nApplicationChooser = 0;

	switch (nApplicationChooser)
	{
	case 0: // Kalman filter
		{
			if (argc < 5)
			{
				printf("video_name sigma threshold patch_size \n");
				//exit;
				return 1;
			}

			float sigma = atof(argv[2]);
			int peak = atoi(argv[3]);
			int patch_size = atoi(argv[4]);
			//int speed = atoi(argv[5]);

			Kalman_Filter kalman(sigma, peak, patch_size); // 1, 1000, 50
			kalman.start_tracking(argv[1]);
		}
		break;

	case 1: //Hough transform
		{
			if (argc < 6)
			{
				printf("Image_Name EdgeLowerThreshold EdgeUpperThreshold VoteThreshold #Bins \n");
				//exit;
				return 1;
			}

			float fEdgeLowerThreshold = atof(argv[2]);
			float fEdgeUpperThreshold = atof(argv[3]);
			int nVoteThreshold = atoi(argv[4]);
			int nBins = atoi(argv[5]);

		    IplImage* img = cvLoadImage( argv[1] );
		    //load gray scale
		    IplImage* grayImg = cvLoadImage(argv[1], 0);

		    //    Filter filter;
		    //    IplImage* result = filter.gaussianFilter(grayImg, 5);
		    //    cvSaveImage("test.png", result, 0);

		    Filter filter;
		    IplImage* edgeImage = filter.myCanny(grayImg, 1, fEdgeLowerThreshold, fEdgeUpperThreshold); // 1 0.1 0.4
		    
			cvSaveImage("edge.png", edgeImage, 0);

		    Hough_Transform hough;
		    IplImage* result = hough.drawHoughLines(grayImg, edgeImage, nVoteThreshold, nBins); // 160 700

			cvSaveImage("result.png", result, 0);
	        cvNamedWindow( "result", CV_WINDOW_NORMAL );
	        cvShowImage( "result", result );
	        cvWaitKey(0);
			cvReleaseImage( &img );
			cvDestroyWindow( "result" );
			cvReleaseImage( &result );
	        cvReleaseImage( &img );
	        cvReleaseImage( &grayImg );
		}
		break;
	}
	




////    Object_Detection objectDetector;
////    IplImage * result = objectDetector.showCorners(img, objectDetector.harris(grayImg, 1.6, 900));
////    cvSaveImage("test.png", result, 0);

//        cvNamedWindow( "InputImage", CV_WINDOW_NORMAL );
//        cvShowImage( "InputImage", img );
//        cvNamedWindow( "result", CV_WINDOW_NORMAL );
//        cvShowImage( "result", result );
//        cvWaitKey(0);
//        cvReleaseImage( &img );
//        cvReleaseImage( &grayImg );
//        cvDestroyWindow( "InputImage" );
//        cvDestroyWindow( "result" );




    ////// feature matching


//    Object_Detection objectDetector;
//    IplImage* img1 = cvLoadImage("im1.png");
//    IplImage* img2 = cvLoadImage("im2.png");

//    printf("load images ok \n");
//    objectDetector.doMatch(img1, img2, 1.0, 10000, 10);//sigma = 1 is better than 2

//    cvNamedWindow( "model", CV_WINDOW_NORMAL );
//    cvNamedWindow( "data", CV_WINDOW_NORMAL );
//    cvShowImage( "model", img1 );
//    cvShowImage( "data", img2 );
//    cvWaitKey(0);
//    cvReleaseImage( &img1 );
//    cvReleaseImage( &img2 );
//    cvDestroyWindow( "model" );
//    cvDestroyWindow( "data" );




}













