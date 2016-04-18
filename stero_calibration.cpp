//
//  stero_calibration.cpp
//  Opencv Test
//
//  Created by Akash Sambrekar on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgcodecs.hpp"
#include<vector>
#include <stdio.h>
#include <string.h>
#include<iostream>

using namespace cv;
using namespace std;

int main(void)
{
 
    Mat left,right;
    
    int nboards=5; // number of different poses
    int board_w=8;   // number of horizontal corners
    int board_h=6;   // number of vertical corners
    
    Size board_sz = Size(board_w, board_h);
    int board_n = board_w*board_h;
    
    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > imagePoints1, imagePoints2;
    vector<Point2f> corners1, corners2;
    
    vector<Point3f> obj;
    for (int j=0; j<board_n; j++)
    {
        obj.push_back(Point3f(j/board_w, j%board_w, 0.0f));
    }
    
    Mat gray1, gray2;
    bool found1=false,found2=false;
    int success=0;
    int i=49;
    int k=0;
    string l = "/Users/akashsambrekar/Desktop/Xcode/Opencv Test/Opencv Test/data/left";
    string r = "/Users/akashsambrekar/Desktop/Xcode/Opencv Test/Opencv Test/data/right";
    string jpg = ".jpg";
    
    while(nboards>success)
    {
        
        left = imread(l+(char)i+jpg,IMREAD_COLOR);
        right = imread(r+(char)i+jpg,IMREAD_COLOR);
        i++;
    
    cvtColor(left, gray1, CV_RGB2GRAY);
    found1 = findChessboardCorners(gray1, board_sz, corners1,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    
    cvtColor(right, gray2, CV_RGB2GRAY);
    found2 = findChessboardCorners(gray2, board_sz, corners2,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    
    cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1),
                 TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
    drawChessboardCorners(gray1, board_sz, corners1, found1);
    imshow("corners_left", gray1);
    
    cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1),
                 TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
    drawChessboardCorners(gray2, board_sz, corners2, found2);
    imshow("corners_right", gray2);
    //waitKey(0);
        
        k = waitKey(10);
        if (found1 && found2)
        {
            k = waitKey(0);
            
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
            printf ("Corners stored\n");
            success++;
            
            if (success >= nboards)
            {
                break;
            }
        }
    
    }
    
    destroyAllWindows();
    
    printf("Starting Calibration\n");
    Mat_<double> CM1; //= Mat(3, 3, CV_64FC1);
    Mat_<double> CM2; //= Mat(3, 3, CV_64FC1);
    Mat_<double> D1, D2;
    Mat_<double> R, T, E, F;
    
    cv::stereoCalibrate(object_points, imagePoints1, imagePoints2, CM1, D1, CM2, D2, left.size(), R, T, E, F,
                    CV_CALIB_FIX_ASPECT_RATIO +
                    CV_CALIB_ZERO_TANGENT_DIST +
                    CV_CALIB_SAME_FOCAL_LENGTH +
                    CV_CALIB_RATIONAL_MODEL +
                    CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5,
                    TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
    
    
    Mat R1, R2, P1, P2, Q;
    stereoRectify(CM1, D1, CM2, D2, left.size(), R, T, R1, R2, P1, P2, Q);
    
    
    Mat map1x, map1y, map2x, map2y;
    Mat imgU1, imgU2, Disparity;
    
    i = 52;
    
    left = imread(l+(char)i+jpg,IMREAD_COLOR);
    right = imread(r+(char)i+jpg,IMREAD_COLOR);

    
    initUndistortRectifyMap(CM1, D1, R1, P1, left.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, right.size(), CV_32FC1, map2x, map2y);
    
    remap(left, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    remap(right, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    
    imshow("Rectify_l",imgU1);
    imshow("Rectify_r", imgU2);
    
    Ptr<StereoBM> smatcher ;
    smatcher->StereoBM::create(0,7);
    smatcher->StereoBM::compute(imgU1,imgU2,Disparity);
    
    imshow("Disp",Disparity);
    
    waitKey(0);

    cout<<"E="<<E<<"\n";
    cout<<"F="<<F<<"\n";
    cout<<"R="<<R<<"\n";
    cout<<"T="<<T<<"\n";
    cout<<"CM1="<<CM1<<"\n";
    cout<<"CM2="<<CM2<<"\n";
    cout<<"D1="<<D1<<"\n";
    cout<<"D2="<<D2<<"\n";
    cout<<"R1="<<R1<<"\n";
    cout<<"R2="<<R2<<"\n";
    cout<<"P1="<<P1<<"\n";
    cout<<"P2="<<P2<<"\n";
    cout<<"Q="<<Q<<"\n";
    return 0;
}