//
//  stereo_matching.cpp
//  Opencv Test
//
//  Created by Akash Sambrekar on 4/12/16.
//  Copyright © 2016 CMU. All rights reserved.
//

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

int main(void)
{
    Mat left  = imread("/Users/akashsambrekar/Downloads/ambush_5_left.jpg" ,IMREAD_COLOR);
    if ( left.empty() )
    {
        cout<<"Cannot read image file: ";
        return -1;
    }
    Mat right = imread("/Users/akashsambrekar/Downloads/ambush_5_right.jpg",IMREAD_COLOR);
    if ( right.empty() )
    {
        cout<<"Cannot read image file: ";
        return -1;
    }
    
    
    Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    Mat conf_map = Mat(left.rows,left.cols,CV_8U);
    conf_map = Scalar(255);
    Rect ROI;
    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;
    
    int wsize = 7;
    int max_disp = 160;
    
    max_disp/=2;
    if(max_disp%16!=0)
        max_disp += 16-(max_disp%16);
    resize(left ,left_for_matcher ,Size(),0.5,0.5);
    resize(right,right_for_matcher,Size(),0.5,0.5);
    
    
    Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
    wls_filter = createDisparityWLSFilter(left_matcher);
    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);
    
    cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
    cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
    
    matching_time = (double)getTickCount();
    left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
    right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
    matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
    
    double lambda = 8000;
    double sigma = 1.5;
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    filtering_time = (double)getTickCount();
    wls_filter->filter(left_disp,left,filtered_disp,right_disp);
    filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    
    conf_map = wls_filter->getConfidenceMap();
    
    // Get the ROI that was used in the last filter call:
    ROI = wls_filter->getROI();
    resize(left_disp,left_disp,Size(),2.0,2.0);
    left_disp = left_disp*2.0;
    ROI = Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
    
    double vis_mult = 1.0;
    
    Mat raw_disp_vis;
    getDisparityVis(left_disp,raw_disp_vis,vis_mult);
    namedWindow("raw disparity", WINDOW_AUTOSIZE);
    imshow("raw disparity", raw_disp_vis);
    Mat filtered_disp_vis;
    getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
    namedWindow("filtered disparity", WINDOW_AUTOSIZE);
    imshow("filtered disparity", filtered_disp_vis);
    waitKey();
    
    return 0;
}

//    Mat g_l,g_r;
//
//    cvtColor(left, g_l, CV_BGR2GRAY);
//    cvtColor(right, g_r, CV_BGR2GRAY);
//    
//    namedWindow( "left", WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "left", left );
//
//    namedWindow( "right", WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "right", right );
//    
//    Mat imgDisparity16S = Mat( left.rows, left.cols, CV_16S );
//    Mat imgDisparity8U = Mat( left.rows, left.cols, CV_8UC1 );
//    
//    int ndisparities = 16*4;
//    int SADWindowSize = 7;
//    
//    Ptr<StereoBM> sbm = StereoBM::create( ndisparities, SADWindowSize );
//    
//    sbm->compute( g_l, g_r, imgDisparity16S );
//
//    double minVal; double maxVal;
//    
//    minMaxLoc( imgDisparity16S, &minVal, &maxVal );
//    
//    printf("Min disp: %f Max value: %f \n", minVal, maxVal);
//    
//    imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
//    
//    Mat right_mask(imgDisparity8U.size(),CV_8UC3);
//    
//    cvtColor(imgDisparity8U, right_mask, CV_GRAY2BGR);
//    
//    Point2f p,q;
//    
//    p.x = 10;p.y = 10;
//    q.x = 80;q.y = 80;
//    
//    arrowedLine(right_mask, p, q, CV_RGB(255,0,0),1.5,8,0,0.2);
//    
//    namedWindow( "windowDisparity", WINDOW_AUTOSIZE );
//    imshow( "windowDisparity", imgDisparity8U );
//    imshow("tp",right_mask);
//    
////    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
////    
////    imshow("disp", disp8);
////
//    waitKey(0);
//    
//    
////    cv::ximgproc::Ptr<DisparityWLSFilter> wls_filter;
//    
//    return 0;
//    
//}