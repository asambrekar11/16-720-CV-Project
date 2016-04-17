//
//  surf.cpp
//  Opencv Test
//
//  Created by Akash Sambrekar on 4/13/16.
//  Copyright © 2016 CMU. All rights reserved.
//

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include<vector>
#include <stdio.h>
#include <string.h>
#include<iostream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;



void StereoMatching(Mat left, Mat right)
{
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
}


static Matx31d NLtriangulation(Point2f P1,
                               Point2f P2,
                               Matx34f M1,
                               Matx34f M2)
{

        
        Matx44d A(P1.x*M1(2,0)-M1(0,0), P1.x*M1(2,1)-M1(0,1), P1.x*M1(2,2)-M1(0,2), P1.x*M1(2,3)-M1(0,3),
                  P1.y*M1(2,0)-M1(1,0), P1.y*M1(2,1)-M1(1,1), P1.y*M1(2,2)-M1(1,2), P1.y*M1(2,3)-M1(1,3),
                  P2.x*M2(2,0)-M2(0,0), P2.x*M2(2,1)-M2(0,1), P2.x*M2(2,2)-M2(0,2), P2.x*M2(2,3)-M2(0,3),
                  P2.y*M2(2,0)-M2(1,0), P2.y*M2(2,1)-M2(1,1), P2.y*M2(2,2)-M2(1,2), P2.y*M2(2,3)-M2(1,3)
                  );
        
        SVD svd(A,cv::SVD::MODIFY_A);
        
        Mat_<double> vt = svd.vt;
        
        Mat_<double> t = vt.row(3);
        
        t = t/t(3);
        
    Matx31d XX(t(0), t(1), t(2));
    
    return XX;
}


Mat GetFeatureDescriptor(Mat image, std::vector<KeyPoint> &keypoints)
{
     Ptr<FeatureDetector> featureDetector = cv::KAZE::create("SIFT");
     featureDetector->detect(image, keypoints);
    
    Mat descriptors;
    Ptr<DescriptorExtractor> featureExtractor = cv::KAZE::create("SIFT");
    featureExtractor->compute(image, keypoints, descriptors);
    
    return descriptors;
    
}

std::vector<DMatch> GetMatches(Mat descriptors1, Mat descriptors2)
{
    Ptr<DescriptorMatcher> bmatcher = cv::DescriptorMatcher::create("BruteForce");
    vector<DMatch> matches;
    bmatcher->match(descriptors1, descriptors2, matches);
    
    return matches;
}

std::vector<DMatch> ComputeGoodMatches(std::vector<DMatch> matches)
{
    double max_dist = 0; double min_dist = 100;
    
    for( int i = 0; i < matches.size(); i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    std::vector<DMatch> good_matches;
    
    for( int i = 0; i < matches.size(); i++ )
    { if( matches[i].distance <= 3*min_dist )
    { good_matches.push_back( matches[i]); }
    }

    return good_matches;
}

void VisualizeMatches(std::vector<DMatch> good_matches,
                      std::vector<KeyPoint> keypoints1,
                      std::vector<KeyPoint> keypoints2,
                      Mat left,
                      Mat right)
{
    Mat img_matches;
    drawMatches(left, keypoints1, right, keypoints2, good_matches, img_matches);
    imshow("matches", img_matches);
}


void GetGoodKeyPoints(std::vector<KeyPoint> keypoints1,
                      std::vector<KeyPoint> keypoints2,
                      std::vector<DMatch> good_matches,
                      std::vector<Point2f> &kps1,
                      std::vector<Point2f> &kps2)
{
    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
        kps1[i] = keypoints1[good_matches[i].queryIdx].pt;
        kps2[i] = keypoints2[good_matches[i].trainIdx].pt;
    }
    
    
}

Matx33f CalculateFundamentalMatrix(std::vector<Point2f> pts1,
                                   std::vector<Point2f> pts2)
{
    
    Matx33f f = findFundamentalMat( pts1, pts2 ,FM_RANSAC, 3, 0.99);
    
    return f;
    
}

Matx33f CalculateEssentialMatrix(Matx33f f, Matx33f K1, Matx33f K2)
{
    Matx33f E;
    
    E = K2.t()*f*K1;
    
    return (Matx33f)E;
}

std::vector<Matx34f> GetProjectionMatrix(Matx33f E)
{
    SVD svd(E,cv::SVD::MODIFY_A);
    
    Mat u = svd.u;
    Mat vt = svd.vt;
    Mat_<double> w = svd.w;
    
    
    double m = (w.at<double>(0,0)+w.at<double>(0,1))/2;
    
    
    w.at<double>(0,0) = m;
    w.at<double>(0,1) = m;
    w.at<double>(0,2) = 0;
    
    Matx33f s(m, 0, 0,
              0, m, 0,
              0, 0, 0);
    
    Matx33f W(0, -1, 0,
              1,  0, 0,
              0,  0, 1);
    
    
    if(determinant(u*Mat(W)*vt)<0)
    {
        W = -W;
    }
    
    Matx34f PrjMtx1(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0);
    
    Mat_<double> R = u*Mat(W)*vt;
    Mat_<double> t = u.col(2);
    
    double min, max;
    minMaxLoc(abs(t),&min,&max);
    
    t = t/(max);
    
    if(t.at<double>(0,1)<0)
    {
        t = -t;
    }
    
    
    Matx34f PrjMatx;
    
    std::vector<Matx34f> PrjMatx2;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                                   R(1,0), R(1,1), R(1,2), t(1),
                                   R(2,0), R(2,1), R(2,2), t(2));
    
    PrjMatx2.push_back(PrjMatx);
    
    
    
    R = u*Mat(W.t())*vt;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                                   R(1,0), R(1,1), R(1,2), t(1),
                                   R(2,0), R(2,1), R(2,2), t(2));
    
    PrjMatx2.push_back(PrjMatx);
    
    return PrjMatx2;
    

}

std::vector<Matx34f> GetFastProjectionMatrix(Matx33f E)
{
    Mat_<double> R1,R2,t;
    
    decomposeEssentialMat(E, R1, R2, t);
    
    double min, max;
    minMaxLoc(abs(t),&min,&max);
    
    t = t/(max);
    
    if(t.at<double>(0,1)<0)
    {
        t = -t;
    }
    
    auto R = R1;
    
    Matx34f PrjMatx;
    
    std::vector<Matx34f> PrjMatx2;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                                   R(1,0), R(1,1), R(1,2), t(1),
                                   R(2,0), R(2,1), R(2,2), t(2));
    
    
    PrjMatx2.push_back(PrjMatx);
    
    R = R2;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                                   R(1,0), R(1,1), R(1,2), t(1),
                                   R(2,0), R(2,1), R(2,2), t(2));
    
    PrjMatx2.push_back(PrjMatx);
    

    return PrjMatx2;
    
}

std::vector<Matx31f> Triangulate(std::vector<Point2f> kps1,
                                 std::vector<Point2f> kps2,
                                 Matx33f K1, Matx33f K2,
                                 Matx34f M1, Matx34f M2)
{
    std::vector<Matx31f> X;
    
    for(int i=0;i<kps1.size();i++)
    {
        Matx31f XX = NLtriangulation(kps1[i], kps2[i], K1*M1, K2*M2);
        X.push_back(XX);
    }
    
    
    for(int i=0;i<X.size();i++)
    {
        cout<<X[i].t()<<"\n";
        
    }
    
    return X;
}


int main(void)
{
    
    //Load Images
    Mat left  = imread("/Users/akashsambrekar/Downloads/im2.png" ,IMREAD_COLOR);
    if ( left.empty() )
    {
        cout<<"Cannot read image file: ";
        return -1;
    }
    Mat right = imread("/Users/akashsambrekar/Downloads/im1.png",IMREAD_COLOR);
    if ( right.empty() )
    {
        cout<<"Cannot read image file: ";
        return -1;
    }
    
    //Feature extraction
    Mat descriptors1,descriptors2;
    std::vector<KeyPoint> keypoints1,keypoints2;
    descriptors1 = GetFeatureDescriptor(left, keypoints1);
    descriptors2 = GetFeatureDescriptor(right, keypoints2);
    
    //Feature Matching
    std::vector<DMatch> matches;
    matches = GetMatches(descriptors1, descriptors2);
    
    std::vector<DMatch> good_matches;
    good_matches = ComputeGoodMatches(matches);
    
    
    //Compute point correspondance wrt good matches
    std::vector<Point2f> kps1(good_matches.size()),kps2(good_matches.size());
    GetGoodKeyPoints(keypoints1, keypoints2, good_matches, kps1, kps2);
    
    
    //Visualise the matches
    VisualizeMatches(good_matches, keypoints1, keypoints2, left, right);
    
    
    //Get Fundamental matrix
    Matx33f f = CalculateFundamentalMatrix(kps1, kps2);

    //Need to find the camera intrinsic paramters of Guidance
    
    
    //Camera Intrinsics. Need to find the camera calibration parameters of Guidance
    Matx33f K1(1520.4, 0, 302.3, 0, 1525.9, 246.9, 0, 0, 1);
    Matx33f K2(1520.4, 0, 302.3, 0, 1525.9, 246.9, 0, 0, 1);
    
    //Get Essential matrix
    Matx33f E = CalculateEssentialMatrix(f, K1, K2);
    
    //Reference frame for first camera
    Matx34f PrjMatx1(1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0);
    
    std::vector<Matx34f> PrjMatx2;
    
    //Get second camera projection matrix
    PrjMatx2 = GetProjectionMatrix(E);
    
    //Triangulate points using least square problem
    std::vector<Matx31f> X = Triangulate(kps1, kps2, K1, K2, PrjMatx1, PrjMatx2[1]);

    //Approach 2 through stereo matching
    StereoMatching(left, right);
    
    waitKey(0);

    
    return 0;

}