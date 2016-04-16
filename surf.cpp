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
#include <opencv2/xfeatures2d/nonfree.hpp>
#include<vector>
#include <stdio.h>
#include <string.h>
#include<iostream>

using namespace cv;
using namespace std;


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

int main(void)
{
    
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
    
    // Create smart pointer for SIFT feature detector.
    Ptr<FeatureDetector> featureDetector1 = cv::KAZE::create("SIFT");
    Ptr<FeatureDetector> featureDetector2 = cv::KAZE::create("SIFT");
    vector<KeyPoint> keypoints1,keypoints2;
    
    
    // Detect the keypoints
    featureDetector1->detect(left, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
    featureDetector2->detect(right, keypoints2); // NOTE: featureDetector is a pointer hence the '->'.
    
    //Similarly, we create a smart pointer to the SIFT extractor.
    Ptr<DescriptorExtractor> featureExtractor1 = cv::KAZE::create("BRIEF");
    Ptr<DescriptorExtractor> featureExtractor2 = cv::KAZE::create("BRIEF");
    
    // Compute the 128 dimension SIFT descriptor at each keypoint.
    // Each row in "descriptors" correspond to the SIFT descriptor for each keypoint
    Mat descriptors1,descriptors2;
    featureExtractor1->compute(left, keypoints1, descriptors1);
    featureExtractor2->compute(right, keypoints2, descriptors2);
    
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    
    double max_dist = 0; double min_dist = 100;
    
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );
    
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<DMatch> good_matches;
    
    for( int i = 0; i < descriptors2.rows; i++ )
    { if( matches[i].distance <= 3.5*min_dist )
    { good_matches.push_back( matches[i]); }
    }
    
    Mat img_matches;
    std::vector<Point2f> kps1(good_matches.size()),kps2(good_matches.size());
    drawMatches(left, keypoints1, right, keypoints2, good_matches, img_matches);
    imshow("matches", img_matches);
    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
        kps1[i] = keypoints1[good_matches[i].queryIdx].pt;
        kps2[i] = keypoints2[good_matches[i].trainIdx].pt;
    }
    
    Matx33f f = findFundamentalMat( kps1, kps2 ,FM_RANSAC, 3, 0.99);
    

    //Need to find the camera intrinsic paramters of Guidance
    
    
    //Camera Intrinsics. Need to find the camera calibration parameters of Guidance
    Matx33f K1(1520.4, 0, 302.3, 0, 1525.9, 246.9, 0, 0, 1);
    Matx33f K2(1520.4, 0, 302.3, 0, 1525.9, 246.9, 0, 0, 1);
    
    //Essential matrix
    
    Matx33f E;
    
    E = K2.t()*f*K1.t();
    
    SVD svd(E,cv::SVD::MODIFY_A);
    
    Mat u = svd.u;
    Mat vt = svd.vt;
    Mat_<double> w = svd.w;
    
    cout<<w<<"\n";

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
    cout<<t<<"\n";
    minMaxLoc(t,&min,&max);
    
    t = t/abs(max);
    
    cout<<t<<"\n";
    
    Matx34f PrjMatx;
    
    std::vector<Matx34f> PrjMatx2;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                            R(1,0), R(1,1), R(1,2), t(1),
                            R(2,0), R(2,1), R(2,2), t(2));
    
    PrjMatx2.push_back(PrjMatx);
    
    t = -t;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                            R(1,0), R(1,1), R(1,2), t(1),
                            R(2,0), R(2,1), R(2,2), t(2));
    
    PrjMatx2.push_back(PrjMatx);
    
    R = u*Mat(W.t())*vt; t=-t;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                            R(1,0), R(1,1), R(1,2), t(1),
                            R(2,0), R(2,1), R(2,2), t(2));
    
    PrjMatx2.push_back(PrjMatx);

    t=-t;
    
    PrjMatx =              Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                            R(1,0), R(1,1), R(1,2), t(1),
                            R(2,0), R(2,1), R(2,2), t(2));
    
    PrjMatx2.push_back(PrjMatx);
    
    
    std::vector<Matx31d> X;
    
    for(int i=0;i<kps1.size();i++)
    {
        Matx31d XX = NLtriangulation(kps1[i], kps2[i], K1*PrjMtx1, K2*PrjMatx2[1]);
        X.push_back(XX);
    }
    
    
    for(int i=0;i<X.size();i++)
    {
        cout<<X[i].t()<<"\n";
        
    }
    
    
     waitKey(0);

    
    return 0;

}