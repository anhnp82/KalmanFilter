#include "object_detection.h"
#include <stdlib.h>
#include <time.h>
#define round(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

Object_Detection::Object_Detection()
{
}



void Object_Detection::
homography_estimation(const std::vector<Hough_Transform::line> & modelCorners
                      , const std::vector<Hough_Transform::line> & dataCorners
                      , const std::vector<matching> & matchings, int k
                      , float homo[][3])
{

    CvMat* A  = cvCreateMat(2*k, 9, CV_32FC1);
    //D has the same dim as A
    CvMat* D  = cvCreateMat(2*k, 9, CV_32FC1);
    CvMat* V  = cvCreateMat(9 , 9, CV_32FC1);
    //CvMat* U  = cvCreateMat(9 , 9, CV_32FC1);

    for (int i = 0; i < k; i++)
    {
        cvmSet(A, 2*i, 0, modelCorners[matchings[i].from].theta); // Set A(i,j)
        cvmSet(A, 2*i, 1, modelCorners[matchings[i].from].rho);
        cvmSet(A, 2*i, 2, 1);
        cvmSet(A, 2*i, 3, 0);
        cvmSet(A, 2*i, 4, 0);
        cvmSet(A, 2*i, 5, 0);
        cvmSet(A, 2*i, 6, -modelCorners[matchings[i].from].theta*dataCorners[matchings[i].to].theta );
        cvmSet(A, 2*i, 7, -modelCorners[matchings[i].from].rho*dataCorners[matchings[i].to].theta );
        cvmSet(A, 2*i, 8, -dataCorners[matchings[i].to].theta);


        cvmSet(A, 2*i+1, 0, 0); // Set A(i,j)
        cvmSet(A, 2*i+1, 1, 0);
        cvmSet(A, 2*i+1, 2, 0);
        cvmSet(A, 2*i+1, 3, modelCorners[matchings[i].from].theta);
        cvmSet(A, 2*i+1, 4, modelCorners[matchings[i].from].rho);
        cvmSet(A, 2*i+1, 5, 1);
        cvmSet(A, 2*i+1, 6, -modelCorners[matchings[i].from].theta*dataCorners[matchings[i].to].rho );
        cvmSet(A, 2*i+1, 7, -modelCorners[matchings[i].from].rho*dataCorners[matchings[i].to].rho );
        cvmSet(A, 2*i+1, 8, -dataCorners[matchings[i].to].rho);
    }

    //cvSVD(A, D, NULL, V, CV_SVD_V_T); // A = U D V^T
    cvSVD(A, D, NULL, V, 0); // A = U D V^T

    float v99 = cvmGet(V, 8, 8); // Get M(i,j)

    homo[0][0] = cvmGet(V, 0, 8)/v99;
    homo[0][1] = cvmGet(V, 1, 8)/v99;
    homo[0][2] = cvmGet(V, 2, 8)/v99;
    homo[1][0] = cvmGet(V, 3, 8)/v99;
    homo[1][1] = cvmGet(V, 4, 8)/v99;
    homo[1][2] = cvmGet(V, 5, 8)/v99;
    homo[2][0] = cvmGet(V, 6, 8)/v99;
    homo[2][1] = cvmGet(V, 7, 8)/v99;
    homo[2][2] = 1;

}




void Object_Detection::
transform(const std::vector<Hough_Transform::line> & modelCorners
          , std::vector<Hough_Transform::line> & dataCorners, int k, float homo[][3]
          , const std::vector<matching> & matchings)
{

    for (int i = 0; i < k; i++)
    {
        float modelPoint[3];
        modelPoint[0] = modelCorners[matchings[i].from].theta;
        modelPoint[1] = modelCorners[matchings[i].from].rho;
        modelPoint[2] = 1;

        float dataPoint[3];

        dataPoint[0] = filter.dot(homo[0], modelPoint, 3);
        dataPoint[1] = filter.dot(homo[1], modelPoint, 3);
        dataPoint[2] = filter.dot(homo[2], modelPoint, 3);

        Hough_Transform::line transformed;
        dataPoint[0] /= dataPoint[2];
        dataPoint[1] /= dataPoint[2];
        transformed.theta = int(dataPoint[0]);
        transformed.rho = int(dataPoint[1]);
        dataCorners.push_back(transformed);
    }

}




void Object_Detection::
get_inliers(const std::vector<Hough_Transform::line> & modelCorners
            , const std::vector<Hough_Transform::line> & dataCorners
            , const std::vector<matching> & matchings, int k
            , float homo[][3]
            , std::vector<matching> & inliers, float squared_dist_threshold)
{

    inliers.clear();
    std::vector<Hough_Transform::line> transformed;
    transform(modelCorners, transformed, k, homo, matchings);

    for (int i = 0; i < k; i++)
    {
        float error_vec[2];
        error_vec[0] = dataCorners[matchings[i].to].theta - transformed[i].theta;
        error_vec[1] = dataCorners[matchings[i].to].rho - transformed[i].rho;

        float squared_norm = error_vec[0]*error_vec[0] + error_vec[1]*error_vec[1];
        if (squared_norm <= squared_dist_threshold)
        {
            inliers.push_back(matchings[i]);
        }
    }
}




void Object_Detection::ransac(const std::vector<Hough_Transform::line> & modelCorners
                              , const std::vector<Hough_Transform::line> & dataCorners
                              , const std::vector<matching> & matchings
                              , int k, float homo[][3]
                              , std::vector<matching> & inliers, int iterations)
{

    /* initialize random seed: */
    srand ( time(NULL) );

    int max_inliers = 0;

    for (int i = 0; i < iterations; i++)
    {

        std::vector<Hough_Transform::line> model_seed, data_seed;
        std::vector<matching> matching_seed;

        //need 4 seeds to estimate homography
        for (int j = 0; j < 4; j++)
        {
            //random number in the range [0, k-1]
            int ran_id = rand() % k;

            model_seed.push_back(modelCorners[matchings[ran_id].from]);
            data_seed.push_back(dataCorners[matchings[ran_id].to]);
            matching ran_match;
            ran_match.from = j;
            ran_match.to = j;
            matching_seed.push_back(ran_match);
        }

        if (!is_good_pose(model_seed, data_seed, matching_seed))
        {
            printf("ill pose \n");
            continue;
        }

        float local_homo[3][3];
        homography_estimation(model_seed, data_seed, matching_seed, 4, local_homo);

        std::vector<matching> local_inliers;
        //error tolarance: 3 pixels
        get_inliers(modelCorners, dataCorners, matchings, k, local_homo, local_inliers, 9);
        if ( (local_inliers.size() > max_inliers) )
        {
            max_inliers = local_inliers.size();
            for(int m = 0; m < 3; m++)
            {
                for(int n = 0; n < 3; n++)
                {
                    homo[m][n] = local_homo[m][n];
                }
            }

            inliers.clear();
            for (int l = 0; l < local_inliers.size(); l++)  inliers.push_back(local_inliers[l]);

        }
    }


}


int Object_Detection::sign(float value)
{
    if(value > 0) return 1;
    if(value < 0) return -1;
    if (value == 0) return 0;
}


bool Object_Detection::
is_good_pose(const std::vector<Hough_Transform::line> & modelCorners
               , const std::vector<Hough_Transform::line> & dataCorners
               , const std::vector<matching> & matchings)
{
    bool good = false;
    int line_map[3]  = {1, 2, 3};
    int opposite1[3] = {2, 1, 2};
    int opposite2[3] = {3, 3, 1};

    float x1f[2] = {modelCorners[matchings[0].from].theta, modelCorners[matchings[0].from].rho};
    float x1t[2] = {dataCorners[matchings[0].to].theta, dataCorners[matchings[0].to].rho};
    for (int i = 0; i < 3; i++)
    {
        float xlf[2] = {modelCorners[matchings[line_map[i]].from].theta
                        , modelCorners[matchings[line_map[i]].from].rho};
        float nf[2] = {xlf[1] - x1f[1], x1f[0] - xlf[0]};

        float xhf[2] = {modelCorners[matchings[opposite1[i]].from].theta
                        , modelCorners[matchings[opposite1[i]].from].rho};

        float xkf[2] = {modelCorners[matchings[opposite2[i]].from].theta
                        , modelCorners[matchings[opposite2[i]].from].rho};

        float phf = nf[0]*(xhf[0] - x1f[0]) + nf[1]*(xhf[1] - x1f[1]);
        float pkf = nf[0]*(xkf[0] - x1f[0]) + nf[1]*(xkf[1] - x1f[1]);

        if (sign(phf) == sign(pkf)) continue;

        float xlt[2] = {dataCorners[matchings[line_map[i]].to].theta
                        , dataCorners[matchings[line_map[i]].to].rho};
        float nt[2] = {xlt[1] - x1t[1], x1t[0] - xlt[0]};

        float xht[2] = {dataCorners[matchings[opposite1[i]].to].theta
                        , dataCorners[matchings[opposite1[i]].to].rho};

        float xkt[2] = {dataCorners[matchings[opposite2[i]].to].theta
                        , dataCorners[matchings[opposite2[i]].to].rho};

        float pht = nt[0]*(xht[0] - x1t[0]) + nt[1]*(xht[1] - x1t[1]);
        float pkt = nt[0]*(xkt[0] - x1t[0]) + nt[1]*(xkt[1] - x1t[1]);

        good = (sign(phf) == sign(pht)) && (sign(pkf) == sign(pkt));


    }

    return good;

}



CvPoint Object_Detection::
doMatch_with_ransac(const std::vector<Hough_Transform::line> & harrisModel,
                    const vector< vector<float> > & despModel,
                    IplImage *data, float sigma, float threshold, int k)
{

    IplImage* dataGray = cvCreateImage( cvGetSize(data), IPL_DEPTH_8U, 1 );
    cvConvertImage(data, dataGray, 0);

    std::vector<Hough_Transform::line> harrisData;
    harrisData = harris(dataGray, sigma, threshold);
    printf("number of corners found in the data: %d \n", harrisData.size());

    vector< vector<float> > despData;
    descriptor_maglap(dataGray, harrisData, 41, sigma, 16, despData);

    std::vector<matching> matchings;
    find_nn_chi2(despModel, despData, matchings);

    partial_sort(matchings.begin(), matchings.begin()+k, matchings.end(), CmpDist());

    float homo[3][3];
    std::vector<matching> inliers;
    //guarantee for 50% of errors
    ransac(harrisModel, harrisData, matchings, k, homo, inliers, 72);

    CvPoint center = draw_object_bbox(harrisData, inliers, data);
    printf("number of inliers %d \n", inliers.size());

    return center;
}





std::vector<CvPoint> Object_Detection::
do_local_match(     const vector< vector<float> > & despModel,
                    IplImage *data, float sigma, float threshold)
{

    IplImage* dataGray = cvCreateImage( cvGetSize(data), IPL_DEPTH_8U, 1 );
    cvConvertImage(data, dataGray, 0);

    std::vector<Hough_Transform::line> harrisData;
    harrisData = harris(dataGray, sigma, threshold);
    printf("number of corners found in the data: %d \n", harrisData.size());

    vector< vector<float> > despData;
    descriptor_maglap(dataGray, harrisData, 11, sigma, 16, despData);

    std::vector<matching> matchings;
    find_nn_chi2(despModel, despData, matchings);

    printf("nn ok \n");

    //partial_sort(matchings.begin(), matchings.begin()+1, matchings.end(), CmpDist());

    if (harrisData.size() > 0) printf("harris %d %d \n", harrisData[0].theta, harrisData[0].rho);
    //printf("harris size %d matching size %d desp size %d \n", harrisData.size(), matchings.size(), despModel.size());

    std::vector<CvPoint> candidates;
    for (int i = 0; i < harrisData.size(); i++)
    {
        CvPoint center;
        center.x = harrisData[i].theta;
        center.y = harrisData[i].rho;
        candidates.push_back(center);
    }

    return candidates;
}







void Object_Detection::
doMatch_with_ransac(IplImage *model, IplImage *data, float sigma, float threshold, int k)
{

    //convert to gray scale
    IplImage* modelGray = cvCreateImage( cvGetSize(model), IPL_DEPTH_8U, 1 );
    //grayscale only has 8-bit depth
    cvConvertImage(model, modelGray, 0);
    IplImage* dataGray = cvCreateImage( cvGetSize(data), IPL_DEPTH_8U, 1 );
    cvConvertImage(data, dataGray, 0);

    std::vector<Hough_Transform::line> harrisModel;
    std::vector<Hough_Transform::line> harrisData;

    harrisModel = harris(modelGray, sigma, threshold);

    printf("number of corners found in the model: %d \n", harrisModel.size());

    harrisData = harris(dataGray, sigma, threshold);

    printf("number of corners found in the data: %d \n", harrisData.size());

    vector< vector<float> > despModel;
    descriptor_maglap(modelGray, harrisModel, 41, sigma, 16, despModel);

    printf("done descriptor 1 \n");

    vector< vector<float> > despData;
    descriptor_maglap(dataGray, harrisData, 41, sigma, 16, despData);

    printf("done descriptor 2 \n");

    std::vector<matching> matchings;
    find_nn_chi2(despModel, despData, matchings);

    int correspondence_no = min(k, (int)harrisModel.size());

    partial_sort(matchings.begin(), matchings.begin()+correspondence_no, matchings.end(), CmpDist());

    showCorners(model, harrisModel, data, harrisData, matchings, correspondence_no);

    float homo[3][3];
    std::vector<matching> inliers;
    //guarantee for 50% of errors
    ransac(harrisModel, harrisData, matchings, correspondence_no, homo, inliers, 72);

    draw_object_bbox(harrisData, inliers, data);
    printf("number of inliers %d \n", inliers.size());

}




CvPoint Object_Detection::
draw_object_bbox(std::vector<Hough_Transform::line> harrisData,
                 std::vector<matching> inliers, IplImage *data)
{
    int minx = data->width;
    int maxx = 0;
    int miny = data->height;
    int maxy = 0;

    for (int i = 0; i < inliers.size(); i++)
    {
        int x = harrisData[inliers[i].to].theta;
        int y = harrisData[inliers[i].to].rho;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }

    cvRectangle (
            data,
            cvPoint(minx, miny),
            cvPoint(maxx, maxy),
            cvScalar(0xff, 0x00, 0x00) /* red */
    );

    return cvPoint((minx + maxx)/2, (miny + maxy)/2);

}





void Object_Detection::doMatch(IplImage *model, IplImage *data, float sigma, float threshold, int k)
{

    //convert to gray scale
    IplImage* modelGray = cvCreateImage( cvGetSize(model), IPL_DEPTH_8U, 1 );
    //grayscale only has 8-bit depth
    cvConvertImage(model, modelGray, 0);
    IplImage* dataGray = cvCreateImage( cvGetSize(data), IPL_DEPTH_8U, 1 );
    cvConvertImage(data, dataGray, 0);

    std::vector<Hough_Transform::line> harrisModel;
    std::vector<Hough_Transform::line> harrisData;

    harrisModel = harris(modelGray, sigma, threshold);

    printf("number of corners found in the model: %d \n", harrisModel.size());

    harrisData = harris(dataGray, sigma, threshold);

    printf("number of corners found in the data: %d \n", harrisData.size());

    vector< vector<float> > despModel;
    descriptor_maglap(modelGray, harrisModel, 41, sigma, 16, despModel);

    printf("done descriptor 1 \n");

    vector< vector<float> > despData;
    descriptor_maglap(dataGray, harrisData, 41, sigma, 16, despData);

    printf("done descriptor 2 \n");

    std::vector<matching> matchings;
    find_nn_chi2(despModel, despData, matchings);
    printf("number of matches %d \n", matchings.size());

    partial_sort(matchings.begin(), matchings.begin()+k, matchings.end(), CmpDist());

    showCorners(model, harrisModel, data, harrisData, matchings, k);

    float homo[3][3];
    homography_estimation(harrisModel, harrisData, matchings, k, homo);

//    printf("homo %f %f %f \n", homo[0][0], homo[0][1], homo[0][2]);
//    printf("homo %f %f %f \n", homo[1][0], homo[1][1], homo[1][2]);
//    printf("homo %f %f %f \n", homo[2][0], homo[2][1], homo[2][2]);

    std::vector<Hough_Transform::line> transformed;
    transform(harrisModel, transformed, k, homo, matchings);

    //printf("transform size %d \n", transformed.size());

    showCorners(data, transformed, matchings);

    for(int i = 0; i < k; i++)
    {
        printf("harris data: %d %d, harris transformed %d %d \n"
               , harrisData[matchings[i].to].theta, harrisData[matchings[i].to].rho
               , transformed[i].theta, transformed[i].rho);
    }

}


std::vector<Hough_Transform::line> Object_Detection::harris(IplImage* img, float sigma, float threshold)
{

    IplImage* imgDx, *imgDy, *imgDx2, *imgDy2, *imgDxy, *tempX, *tempY, *tempXY;

    imgDx = filter.imageXDerivative(img, sigma);
    imgDy = filter.imageYDerivative(img, sigma);

    tempX = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(tempX);
    tempY = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(tempY);
    tempXY = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(tempXY);

    BwImageFloat wrapImgDx(imgDx);
    BwImageFloat wrapImgDy(imgDy);
    BwImageFloat wrapTempX(tempX);
    BwImageFloat wrapTempY(tempY);
    BwImageFloat wrapTempXY(tempXY);

    float sigma2 = sigma*sigma;

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {

            wrapTempX[i][j] = sigma2*wrapImgDx[i][j]*wrapImgDx[i][j];
            wrapTempY[i][j] = sigma2*wrapImgDy[i][j]*wrapImgDy[i][j];
            wrapTempXY[i][j] = sigma2*wrapImgDx[i][j]*wrapImgDy[i][j];

        }
    }

    imgDx2 = filter.gaussianFilter(tempX, 1.6*sigma, true);
    imgDy2 = filter.gaussianFilter(tempY, 1.6*sigma, true);
    imgDxy = filter.gaussianFilter(tempXY, 1.6*sigma, true);

    //cvSaveImage("test.png", imgDxy, 0);


    IplImage* imgDet, *imgTrace;
    imgDet = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(imgDet);
    imgTrace = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(imgTrace);

    BwImageFloat wrapImgDet(imgDet);
    BwImageFloat wrapImgTrace(imgTrace);
    BwImageFloat wrapImgDx2(imgDx2);
    BwImageFloat wrapImgDy2(imgDy2);
    BwImageFloat wrapImgDxy(imgDxy);

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {

            wrapImgDet[i][j] = wrapImgDx2[i][j]*wrapImgDy2[i][j] - wrapImgDxy[i][j]*wrapImgDxy[i][j];
            wrapImgTrace[i][j] = wrapImgDx2[i][j] + wrapImgDy2[i][j];

        }
    }

    // Corner response function
    float alpha = 0.06;

    IplImage* imgPts;
    imgPts = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
    cvZero(imgPts);

    BwImageFloat wrapImgPts(imgPts);

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {

            wrapImgPts[i][j] = wrapImgDet[i][j] - alpha*sigma*wrapImgTrace[i][j]*wrapImgTrace[i][j];

        }
    }

    std::vector<Hough_Transform::line> corners = hough.findhoughpeaks(imgPts, threshold);

    cvReleaseImage( &imgDx );
    cvReleaseImage( &imgDy );
    cvReleaseImage( &imgDx2 );
    cvReleaseImage( &imgDy2 );
    cvReleaseImage( &imgDxy );
    cvReleaseImage( &tempX );
    cvReleaseImage( &tempY );
    cvReleaseImage( &tempXY );
    cvReleaseImage( &imgDet );
    cvReleaseImage( &imgTrace );
    cvReleaseImage( &imgPts );

    //printf("number of corners found: %d \n", corners.size());

    return corners;

}


IplImage * Object_Detection::
showCorners(IplImage *img, const std::vector<Hough_Transform::line> & corners
            , const std::vector<matching> & matchings)
{
    IplImage* result = cvCloneImage( img );
    //cvCopyImage( img, result );
	//cvCopy( img, result );

    for (int i = 0; i < corners.size(); i++)
    {

        CvPoint harris = cvPoint(corners[i].theta, corners[i].rho);

        CvPoint indexPos = cvPoint(corners[i].theta - 5, corners[i].rho - 5);
        //replace img by result
        cvCircle(img, harris, 1, cvScalar(0,255,0), 1);
        string index = convertInt(matchings[i].from);
        CvFont font;
        double hScale = 0.3;
        double vScale = 0.3;
        int    lineWidth = 1;
        cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
        //replace img by result
        cvPutText(img, index.c_str(), indexPos, &font, cvScalar(255, 0, 0));

    }

    return result;
}




void Object_Detection::showCorners(IplImage *model, const std::vector<Hough_Transform::line> & modelCorners
                                   , IplImage * data, const std::vector<Hough_Transform::line> & dataCorners
                                   , const std::vector<matching> & matchings, int k)
{


    for (int i = 0; i < k; i++)
    {

        CvPoint harris = cvPoint(modelCorners[matchings[i].from].theta, modelCorners[matchings[i].from].rho);
        CvPoint indexPos = cvPoint(modelCorners[matchings[i].from].theta + 2, modelCorners[matchings[i].from].rho + 2);

        cvCircle(model, harris, 1, cvScalar(0,255,0), 1);

        string index = convertInt(matchings[i].from);

        CvFont font;
        double hScale = 0.3;
        double vScale = 0.3;
        int    lineWidth = 1;
        cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
        cvPutText(model, index.c_str(), indexPos, &font, cvScalar(0,255,0));

        //the second one
        CvPoint harris2 = cvPoint(dataCorners[matchings[i].to].theta, dataCorners[matchings[i].to].rho);
        CvPoint indexPos2 = cvPoint(dataCorners[matchings[i].to].theta + 2, dataCorners[matchings[i].to].rho + 2);

        cvCircle(data, harris2, 1, cvScalar(0,255,0), 1);

        printf("distance %f \n", matchings[i].distance);

        cvPutText(data, index.c_str(), indexPos2, &font, cvScalar(0,255,0));

    }

}



string Object_Detection::convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}


float Object_Detection::hist_dist_chi2(const vector<float> & h1, const vector<float> & h2)
{

    float dist = 0;

    for (int i = 0; i < h1.size(); i++)
    {
        float diff2 = (h1[i] - h2[i])*(h1[i] - h2[i]);
        float sum = (h1[i] + h2[i]);

        //printf("hist 1 %f hist 2 %f \n", h1[i], h2[i]);

        if (sum > 0)
        {
            dist += diff2/sum;
        }
    }

    //printf("dist %f \n", dist);

    return dist;
}



void Object_Detection::
find_nn_chi2(const std::vector< vector<float> > & desp1, const std::vector< vector<float> > & desp2
             , std::vector<matching> & matchings)
{

    //#pragma omp parallel for
    for (int i = 0; i < desp1.size(); i++)
    {
        matching match;
        match.distance = numeric_limits<float>::max();
        match.to = 0;
        match.from = i;

        for (int j = 0; j < desp2.size(); j++)
        {
            float dist = hist_dist_chi2(desp1[i], desp2[j]);

            if (dist < match.distance)
            {
                match.distance = dist;
                match.to = j;
            }
        }

        matchings.push_back(match);

    }
}


IplImage* Object_Detection::subImage(IplImage* img, int minx, int miny, int maxx, int maxy)
{
    IplImage* win = cvCreateImage( cvSize(maxx - minx + 1, maxy - miny + 1), IPL_DEPTH_8U, 1 );
    cvZero(win);
    BwImage wrapWin(win);
    BwImage wrapImg(img);

    for (int i = 0; i < win->height; i++)
    {
        for (int j = 0; j < win->width; j++)
        {
            wrapWin[i][j] = wrapImg[i+miny][j+minx];
        }
    }


    return win;

}


void Object_Detection::
descriptor_maglap(IplImage *img, const std::vector<Hough_Transform::line> & corners
                  , int size, float sigma, int bins, std::vector< vector<float> > & desp)
{
    int rad = round((size-1)/2);
    int h = img->height;
    int w = img->width;


    //parallelize this block produces different result than sequential code
    //#pragma omp parallel for
    for (int i = 0; i < corners.size(); i++)
    {
        int minx = max(corners[i].theta - rad, 0);
        int maxx = min(corners[i].theta + rad, w - 1);
        int miny = max(corners[i].rho - rad, 0);
        int maxy = min(corners[i].rho + rad, h - 1);

//        cvSetImageROI( img, cvRect(minx, miny, maxx, maxy) );
//        /* create destination image
//           Note that cvGetSize will return the width and the height of ROI */
//        IplImage* window = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
//        cvZero(window);
//        /* copy subimage */
//        cvCopy(img, window, NULL);

//        /* always reset the Region of Interest */
//        cvResetImageROI(img);

		//printf("roi min(%d, %d) max(%d, %d) \n", minx, miny, maxx, maxy);

        IplImage* window = subImage(img, minx, miny, maxx, maxy);
        //IplImage* window = subImage(img, 0, 0, 40, 40);

        vector<float> hist(bins*bins);

        magLapHist(window, sigma, bins, hist);

        cvReleaseImage( &window );

        desp.push_back(hist);
    }
}


void Object_Detection::magLapHist(IplImage* img, float sigma, int bins, vector<float> & h)
{

    IplImage* imgMag = filter.gradMag(img, sigma);
    IplImage* imgLap = filter.laplace(img, sigma);

    BwImageFloat wrapMag(imgMag);
    BwImageFloat wrapLap(imgLap);

    //quantize the images to "bins" number of values
    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {

            //printf("mag %f lap %f \n", wrapMag[i][j], wrapLap[i][j]);

            //fix range [0, 100] for gradient magnitude
            if (wrapMag[i][j] > 100) wrapMag[i][j] = 100;
            wrapMag[i][j] = floor(wrapMag[i][j]*(bins/101.0f));

            //shift range of laplacian by 60
            //fix range [0, 119] for laplacian
            wrapLap[i][j] += 60;
            if (wrapLap[i][j] < 0) wrapLap[i][j] = 0;
            if (wrapLap[i][j] > 119) wrapLap[i][j] = 119;
            wrapLap[i][j] = floor(wrapLap[i][j]*(bins/120.0f));

            //printf("mag %f lap %f \n", wrapMag[i][j], wrapLap[i][j]);

        }
    }

    for (int i = 0; i < bins*bins; i++) h[i] = 0;

    float sum = 0;

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {
            //increment a histogram bin which corresponds to the value
            //of pixel i,j;
            int mag = wrapMag[i][j];
            int lap = wrapLap[i][j];

            if (mag*bins + lap > bins*bins) printf("mag %d lap %d \n", mag, lap);

            h[mag*bins + lap] += 1;
            sum++;
        }
    }

    //normalize the histogram such that its integral (sum) is equal 1
    if (sum > 0)
    {
        for (int i = 0; i < bins*bins; i++)
        {
            h[i] /= sum;
            if (h[i] < 0) printf("normalized hist %f \n", h[i]);
            //printf("normalized hist %f \n", h[i]);
        }
        //printf("sum hist %f \n", sum);
    }

    //printf("done hist \n");

    cvReleaseImage( &imgMag );
    cvReleaseImage( &imgLap );

}
