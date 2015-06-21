#include "kalman_filter.h"
#include "object_detection.h"
#include <iostream>
#include <limits>

bool Kalman_Filter::drawing_box = false;
CvRect Kalman_Filter::box = cvRect(-1,-1,0,0);

Kalman_Filter::Kalman_Filter(float sigma, float threshold, float patch_size)
{

    this->sigma = sigma;
    this->threshold = threshold;
    this->patch_size = patch_size;

    dt = 1/25;

    //Dynamics model update matrix
    D = Mat::eye(4, 4, CV_32FC1);

    D.at<float>(0, 2) = dt;
    D.at<float>(1, 3) = dt;

    //Observation conversion matrix/ measurement matrix
    M = Mat::zeros(2, 4, CV_32FC1);
    M.at<float>(0, 0) = 1;
    M.at<float>(1, 1) = 1;

    I = Mat::eye(4, 4, CV_32FC1);

    sigma_d = I*1.78;

    sigma_m = Mat::eye(2, 2, CV_32FC1)*1.78;

    sigma_predict = I*1.78;

    x_predict = Mat::zeros(4, 1, CV_32FC1);

}


Kalman_Filter::~Kalman_Filter()
{

}



void Kalman_Filter::draw_box( IplImage* img, CvRect rect )
{
    cvRectangle (
            img,
            cvPoint(rect.x, rect.y),
            cvPoint(rect.x + rect.width, rect.y + rect.height),
            cvScalar(0xff, 0x00, 0x00) /* red */
    );
}



// This is our mouse callback. If the user
// presses the left button, we start a box.
// when the user releases that button, then we
// add the box to the current image. When the
// mouse is dragged (with the button down) we
// resize the box.
//
void Kalman_Filter::my_mouse_callback(
        int event, int x, int y, int flags, void* param)
{
    IplImage* image = (IplImage*) param;
    switch( event )
    {
        case CV_EVENT_MOUSEMOVE:
        {
            if( drawing_box )
            {
                box.width = x-box.x;
                box.height = y-box.y;
            }
        }
        break;

        case CV_EVENT_LBUTTONDOWN:
        {
            drawing_box = true;
            box = cvRect(x, y, 0, 0);
        }
        break;

        case CV_EVENT_LBUTTONUP:
        {
            drawing_box = false;
            if(box.width<0)
            {
                box.x+=box.width;
                box.width *=-1;
            }
            if(box.height<0)
            {
                box.y+=box.height;
                box.height*=-1;
            }

            //do not contaminate the original image here
            //draw_box(image, box);
        }
        break;

    }
}



CvRect Kalman_Filter::get_roi_from_user(IplImage *img)
{
    IplImage* temp = cvCloneImage( img );

    cvNamedWindow( "Box Example" );
    // Here is the crucial moment that we actually install
    // the callback. Note that we set the value ‘param’ to
    // be the image we are working with so that the callback
    // will have the image to edit.
    //
    cvSetMouseCallback(
                "Box Example",
                Kalman_Filter::my_mouse_callback,
                (void*) img
    );

    // The main program loop. Here we copy the working image
    // to the ‘temp’ image, and if the user is drawing, then
    // put the currently contemplated box onto that temp image.
    // display the temp image, and wait 15ms for a keystroke,
    // then repeat...
    //
    while( 1 )
    {
        //cvCopyImage( img, temp );
		//cvCopy( img, temp );
        if( drawing_box ) draw_box( temp, box );
        cvShowImage( "Box Example", temp );
        if( cvWaitKey( 15 )==27 ) break;
    }
    // Be tidy
    //
    cvReleaseImage( &temp );
    cvDestroyWindow( "Box Example" );
    return box;

}



void Kalman_Filter::video_extraction()
{
    cvNamedWindow( "video", CV_WINDOW_AUTOSIZE );
    CvCapture* capture = cvCreateFileCapture( "fi-br-m1.avi" );
    IplImage* frame;

    int frameNo = 0;

    while(1)
    {
        frame = cvQueryFrame( capture );

        if (frameNo == 0) cvSaveImage("model.png", frame, 0);

        if (frameNo == 30) cvSaveImage("data.png", frame, 0);

        if( !frame ) break;
        cvShowImage( "video", frame );
        char c = cvWaitKey(33);
        if( c == 27 ) break;

        frameNo++;
    }
    cvReleaseCapture( &capture );
    cvDestroyWindow( "video" );

}





void Kalman_Filter::start_tracking(const char * file_name)
{
    cvNamedWindow( "video", CV_WINDOW_AUTOSIZE );
    CvCapture* capture = cvCreateFileCapture( file_name );

    IplImage * model;
    CvRect roi;

    Object_Detection objectDetector;

    std::vector<Hough_Transform::line> harrisModel;
    std::vector<Hough_Transform::line> features;
    vector< vector<float> > despModel;

    IplImage* frame;
    int frameNo = 0;

    while(1)
    {
        frame = cvQueryFrame( capture );

        if (frameNo == 0)
        {
            model = get_model(frame, roi);
            //convert to gray scale
            IplImage* modelGray = cvCreateImage( cvGetSize(model), IPL_DEPTH_8U, 1 );
            //grayscale only has 8-bit depth
            cvConvertImage(model, modelGray, 0);

            harrisModel = objectDetector.harris(modelGray, sigma, threshold);
            printf("number of corners found in the model: %d \n", harrisModel.size());
            objectDetector.descriptor_maglap(modelGray, features, 41, sigma, 16, despModel);

            int maxId = 0;
            float maxPeak = -1;
            for (int i = 0; i < harrisModel.size(); i++)
            {
                if (harrisModel[i].peak > maxPeak)
                {
                    maxPeak = harrisModel[i].peak;
                    maxId = i;
                }
            }

            Hough_Transform::line feature;
            feature.theta = harrisModel[maxId].theta + roi.x;
            feature.rho = harrisModel[maxId].rho + roi.y;

            features.push_back(feature);
            x_predict.at<float>(0, 0) = feature.theta;
            x_predict.at<float>(1, 0) = feature.rho;
            m_trajectory.push_back(cvPoint(feature.theta, feature.rho));



            printf("feature no %d peak %f position %d %d \n", maxId, maxPeak, feature.theta, feature.rho);
        }

		if( !frame ) break;

        if (frameNo > 0)
        {
            track_object(frame, despModel);
        }

        cvShowImage( "video", frame );
        char c = cvWaitKey(10);
        //if( c == 27 ) break;

        frameNo++;
    }
    cvReleaseCapture( &capture );
    cvDestroyWindow( "video" );

}




void Kalman_Filter::track_object(IplImage *img, vector< vector<float> > despModel
                                 )
{
    printf("begin track \n");

    //associate
    //find feature within the ellipse centered at x_predict with shape sigma_predict

    CvMat* shape  = cvCreateMat(2,2,CV_32FC1);
    CvMat* eigenVec  = cvCreateMat(2,2,CV_32FC1);
    CvMat* eigenVal  = cvCreateMat(2,1,CV_32FC1);
    cvmSet(shape, 0, 0, sigma_predict.at<float>(0, 0));
    cvmSet(shape, 0, 1, sigma_predict.at<float>(0, 1));
    cvmSet(shape, 1, 0, sigma_predict.at<float>(1, 0));
    cvmSet(shape, 1, 1, sigma_predict.at<float>(1, 1));

    cvEigenVV(shape, eigenVec, eigenVal, 1e-15);

    //cvEigenVV(&A, &E, &l);  // l = eigenvalues of A (descending order)
                              // E = corresponding eigenvectors (rows)

    float radius = ceil(3*sqrt(cvmGet(eigenVal,0,0)));
    radius += patch_size;

    CvRect roi;
    float minx = max(x_predict.at<float>(0, 0) - radius, 0.0f);
    float miny = max(x_predict.at<float>(1, 0) - radius, 0.0f);
    float maxx = min(x_predict.at<float>(0, 0) + radius, (float)(img->width-1));
    float maxy = min(x_predict.at<float>(1, 0) + radius, (float)img->height-1);
    roi.x = minx;
    roi.y = miny;
    roi.width = maxx-minx;
    roi.height = maxy-miny;

    printf("roi: %d %d %d %d \n", roi.x, roi.y, roi.width, roi.height);

    cvSetImageROI(img, roi);
    /* create destination image
       Note that cvGetSize will return the width and the height of ROI */
    IplImage* search_region = cvCreateImage( cvGetSize(img), img->depth, img->nChannels );
    cvZero(search_region);
    /* copy subimage */
    cvCopy(img, search_region, NULL);

    /* always reset the Region of Interest */
    cvResetImageROI(img);

    printf("ellipse \n");

    Object_Detection objectDetector;

    std::vector<CvPoint> measurements = objectDetector.do_local_match(despModel, search_region, sigma, threshold);
    float minDist = FLT_MAX;
    int minIndex = INT_MAX;
    CvMat* covMat  = cvCreateMat(2,2,CV_32FC1);
    CvMat* predictedPoint  = cvCreateMat(1,2,CV_32FC1);
    cvmSet(covMat, 0, 0, sigma_predict.at<float>(0, 0));
    cvmSet(covMat, 0, 1, sigma_predict.at<float>(0, 1));
    cvmSet(covMat, 1, 0, sigma_predict.at<float>(1, 0));
    cvmSet(covMat, 1, 1, sigma_predict.at<float>(1, 1));
    cvmSet(predictedPoint, 0, 0, x_predict.at<float>(0, 0));
    cvmSet(predictedPoint, 0, 1, x_predict.at<float>(1, 0));
    for(int i = 0; i < measurements.size(); i++)
    {
        CvMat* hypothesis  = cvCreateMat(1,2,CV_32FC1);

        cvmSet(hypothesis, 0, 0, measurements[i].x+minx);
        cvmSet(hypothesis, 0, 1, measurements[i].y+miny);
        CvMat* inverted = cvCreateMat(2, 2, CV_32FC1);
        cvInvert( covMat, inverted, CV_LU);
        float distance = cvMahalanobis( predictedPoint, hypothesis, inverted);
        if(minDist > distance)
        {
            minDist = distance;
            minIndex = i;
        }
    }

	CvPoint measurement = {roi.width/2, roi.height/2};
	if (minIndex != INT_MAX && measurements.size() > 0) measurement = measurements[minIndex];
    measurement.x += minx;
    measurement.y += miny;

    //correction
    Mat temp_invert = M * sigma_predict * M.t() + sigma_m;
    Mat K = sigma_predict * M.t() * temp_invert.inv();

    Mat y = Mat::zeros(2, 1, CV_32FC1);
    y.at<float>(0, 0) = measurement.x;
    y.at<float>(1, 0) = measurement.y;

    Mat x_correct = x_predict + K * (y - M * x_predict);
    Mat sigma_correct = (I - K * M) * sigma_predict;

    //prediction
    x_predict = D * x_correct;
    sigma_predict = D * sigma_correct * D.t() + sigma_d;

	std::cout << "x_correct: \n" << x_correct << std::endl;

    m_trajectory.push_back(cvPoint(x_correct.at<float>(0, 0), x_correct.at<float>(1, 0)));
    for (int i = 0; i < m_trajectory.size(); i++)
    {
        cvCircle(img, m_trajectory[i], 3, cvScalar(0,255,0), 2);
    }
}



IplImage* Kalman_Filter::get_model(IplImage * img1, CvRect & roi)
{

    roi = get_roi_from_user(img1);
    //printf("roi: %d %d %d %d \n", roi.x, roi.y, roi.width, roi.height);

    cvSetImageROI(img1, roi);
    /* create destination image
       Note that cvGetSize will return the width and the height of ROI */
    IplImage* model = cvCreateImage( cvGetSize(img1), img1->depth, img1->nChannels );
    cvZero(model);
    /* copy subimage */
    cvCopy(img1, model, NULL);

    /* always reset the Region of Interest */
    cvResetImageROI(img1);

    return model;

}




void Kalman_Filter::extract_measurement(const char * model_file, const char * data_file
                                        , float sigma, float threshold, int k)
{
    IplImage *img1 = cvLoadImage(model_file);
    IplImage *img2 = cvLoadImage(data_file);
    CvRect roi = get_roi_from_user(img1);
    printf("roi: %d %d %d %d \n", roi.x, roi.y, roi.width, roi.height);

    cvSetImageROI(img1, roi);
    /* create destination image
       Note that cvGetSize will return the width and the height of ROI */
    IplImage* model = cvCreateImage( cvGetSize(img1), img1->depth, img1->nChannels );
    cvZero(model);
    /* copy subimage */
    cvCopy(img1, model, NULL);

    /* always reset the Region of Interest */
    cvResetImageROI(img1);

    Object_Detection objectDetector;

    //sigma and k vary for each dataset
    //still need geometric verification, thus need enough k i.e. 20
    //k=30 gives too many errors
    //running in real time is fancy and not realistic
    //objectDetector.doMatch(model, img2, 2.0, 100000, 10);//20
    objectDetector.doMatch_with_ransac(model, img2, sigma, threshold, k);//20

    cvNamedWindow( "model", CV_WINDOW_NORMAL );
    cvNamedWindow( "data", CV_WINDOW_NORMAL );
    cvShowImage( "model", model );
    cvShowImage( "data", img2 );
    cvWaitKey(0);
    cvReleaseImage( &img1 );
    cvReleaseImage( &img2 );
    cvReleaseImage( &model );
    cvDestroyWindow( "model" );
    cvDestroyWindow( "data" );
}















