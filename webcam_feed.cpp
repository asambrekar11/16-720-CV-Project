//
//  webcam_feed.cpp
//  Opencv Test
//
//  Created by Akash Sambrekar on 4/18/16.
//  Copyright © 2016 CMU. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include<vector>
#include<time.h>
#include <stdio.h>
#include <string.h>
#include<iostream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

vector<Point2f> prevleftpts, prevrightpts;
Mat prevleftimg, prevrightimg;
Point2f z_leftprev,z_leftnext,z_rightprev,z_rightnext;

#define MIN_FEATURES (uint) 100
#define FEATURE_THRESHOLD (uint) 400
#define MAX_FEATURES (uint) 900
#define QUALITY_EIGVALUE (double) 0.01
#define MIN_DISTANCE (double) 5
#define OF_MARGIN (double) 0.5

#define WIDTH 640
#define HEIGHT 360

Mat     	g_greyscale_image_left;
Mat 		g_greyscale_image_right;

Matx33f K1(1873.78172, 0, 730.2225,0, 1873.78172, 664.92407,0, 0, 1);
Matx33f K2(1873.7817, 0, 882.4448, 0, 1873.7817, 682.70654, 0, 0, 1);

Matx33f E(-0.005338900188870516, 0.09581321279386693, -0.07103465696407894,
          -0.2197756515487002, 0.04344572998251601, -2.615114438699082,
          0.02122976293084764, 2.623537383965801, 0.04443663852323315);

Matx34f M1(1,0,0,0,
           0,1,0,0,
           0,0,1,0);

Matx44f Q(1, 0, 0, 1148.422351837158,
           0, 1, 0, 2968.675243377686,
           0, 0, 0, 1315.680854778361,
           0, 0, -0.3808502719556529, 0);



Scalar GiveColor(int k)
{
    if(k == 0)
    {
        return CV_RGB(255,0,0);
        
    }else if(k == 1)
    {
        return CV_RGB(0,255,0);
        
    }else if(k == 2)
    {
        return CV_RGB(0,0,255);
        
    }else if(k == 3)
    {
        return CV_RGB(255,0,255);
    }else if(k == 4)
    {
        return CV_RGB(255,255,0);
    }else
    {
        return CV_RGB(104,200,134);
    }
}




int GetClosetPoint(Point2f a,std::vector<Point2f> centres)
{
    double minD = 10^6;
    int label=0;
    for(auto i=0;i<centres.size();i++)
    {
        if(norm(a-centres[i])<minD)
        {
            label = i;
            minD = norm(a-centres[i]);
        }
    }
    
    return label;
}

int *KmeanClustering(std::vector<Point2f> kpts, int ClusterNum, int itr)
{
    int *labels = new int [kpts.size()];
    srand((int)time(nullptr));
    
    std::vector<Point2f> centres;
    std::vector<std::vector<Point2f>> update;

    centres.resize(ClusterNum);
    
    for(int i=0;i<ClusterNum;i++)
    {
        int a = round(rand()%kpts.size());
        cout<<a<<"\n";
        centres[i] = kpts[a];
    }
    
    for(int i=0;i<2;i++)
    {
        
        for(int j=0;j<kpts.size();j++)
        {
            Point2f r = kpts[j];
            labels[j] = GetClosetPoint(r, centres);
        }
        
        for(int j=0;j<ClusterNum;j++)
        {
            std::vector<Point2f> cluster;
            for(int k=0;k<kpts.size();k++)
            {
                if(labels[k]==j)
                {
                    cluster.push_back(kpts[k]);
                }
            }
            update.push_back(cluster);
        }
        
        for(int j=0;j<ClusterNum;j++)
        {
            auto todo = update.back();
            update.pop_back();
            double meanx = sum(todo)[0]/todo.size();
            double meany = sum(todo)[1]/todo.size();
            Point2f meanP(meanx,meany);
            centres[j] = meanP;
        }
        
    }
    
    
    
    return labels;
    
}


Mat StereoMatching(Mat left, Mat right)
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
    imshow("raw disparity", raw_disp_vis);
    moveWindow("raw disparity", 0, 400);
    
    Mat filtered_disp_vis;
    getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
    normalize(filtered_disp_vis, filtered_disp_vis, 0, 255, CV_MINMAX, CV_8U);
    
    imshow("filtered disparity", filtered_disp_vis);
    moveWindow("raw disparity", 650, 400);
    
    return raw_disp_vis;
}



int main(void)
{
    int k=0;
    int frame_num=0;
    cv::Mat mean_;
    
    
    //clustering
    int clusterNum = 3;
    int attempts = 10;
    Mat labels1,labels2;
    Mat_<double> centres1,centres2;
    
    
    TermCriteria TC;
    TC = TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
    
    Mat img1, img2;
    CvCapture *capture1 = cvCaptureFromCAM(1);
    CvCapture *capture2 = cvCaptureFromCAM(2);
    
    double t,t1=0;
    
    z_leftprev.x = 0;
    z_leftprev.y = 0;
    
    z_rightprev.x = 0;
    z_rightprev.y = 0;
    
    while(1)
    {
        
        IplImage* frame1 = cvQueryFrame( capture1 );
        IplImage* frame2 = cvQueryFrame( capture2 );
        
        if(capture1!=NULL||capture2!=NULL)
        {
            t = double(getTickCount());
            img1 = cvarrToMat(frame1);
            img2 = cvarrToMat(frame2);
            
            string ctrl_text = "";
        
            resize(img1,img1,Size(WIDTH,HEIGHT),CV_INTER_CUBIC);
            resize(img2,img2,Size(WIDTH,HEIGHT),CV_INTER_CUBIC);
            
            cvtColor(img1, g_greyscale_image_left, CV_RGB2GRAY);
            cvtColor(img2, g_greyscale_image_right, CV_RGB2GRAY);
        
        
            vector<Point2f> nextleftpts,kpts1,vkpts1;
            if(prevleftpts.size()<MIN_FEATURES)
            {
                ctrl_text = "STOP MOVING";

            //control input
            }
            else if(frame_num>0)
            {
            t = (double)getTickCount();
            vector<uchar> status;
            vector<float> error;
            vector<Mat> nextleftpyr,prevleftpyr;
            buildOpticalFlowPyramid(prevleftimg, prevleftpyr, Size(7,7), 4);
            buildOpticalFlowPyramid(g_greyscale_image_left, nextleftpyr, Size(7,7), 4);
            CvTermCriteria optical_flow_termination_criteria
            = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.1 );
            calcOpticalFlowPyrLK(prevleftpyr, nextleftpyr, prevleftpts, nextleftpts, status, error,Size(7,7),4,optical_flow_termination_criteria,OPTFLOW_LK_GET_MIN_EIGENVALS);
            
            Mat left_mask1(g_greyscale_image_left.size(),CV_8UC3);
            Mat left_mask2(g_greyscale_image_left.size(),CV_8UC3);
            cvtColor(g_greyscale_image_left, left_mask1, CV_GRAY2BGR);
            cvtColor(g_greyscale_image_left, left_mask2, CV_GRAY2BGR);
            
                ctrl_text = "GO";
                
            for(size_t i=0;i<nextleftpts.size();i++)
            {
                if(nextleftpts[i].x>=0 && nextleftpts[i].x<=WIDTH && nextleftpts[i].y>=0 && nextleftpts[i].y<=HEIGHT)
                {
                    CvScalar line_color; line_color = CV_RGB(255,0,0);
                    Point2f p,q;
                    p.x = (double) prevleftpts[i].x;
                    p.y = (double) prevleftpts[i].y;
                    q.x = (double) nextleftpts[i].x;
                    q.y = (double) nextleftpts[i].y;
                    
                    double d = sqrt(pow(p.x-q.x,2)+pow(p.y-q.y,2));
                    
                    if(d<20)
                    {
                        //circle(left_mask, q, 4, Scalar(0,255,0),-1);
                        arrowedLine(left_mask1, p, q, Scalar(255,100,100) , 1, CV_AA, 0, 0.4);
                        arrowedLine(left_mask2, p, q, Scalar(255,100,100) , 1, CV_AA, 0, 0.4);
                    }
                    
                    
                        vkpts1.push_back((q-p)*getTickFrequency()/(t-t1));
                        kpts1.push_back(nextleftpts[i]);
                    
                }
            }
               
                
               
                if(kpts1.size()!=0)
                {
                    kmeans(kpts1, clusterNum, labels1, TC, attempts,  KMEANS_PP_CENTERS, centres1);
                    kmeans(vkpts1, clusterNum, labels2, TC, attempts,  KMEANS_PP_CENTERS, centres2);
                    
                    reduce(centres1, mean_, 0, CV_REDUCE_AVG);
                    z_leftnext.x = mean_.at<double>(0,0);
                    z_leftnext.y = mean_.at<double>(0,1);
                    
                    if(z_leftnext.x>z_leftprev.x&&norm(z_leftprev-z_leftnext)>10)
                    {
                        ctrl_text += " RIGHT";
                    }
                    if(z_leftnext.x<z_leftprev.x&&norm(z_leftprev-z_leftnext)>10)
                    {
                        ctrl_text += " LEFT";
                    }
                    if(z_leftnext.y>z_leftprev.y&&norm(z_leftprev-z_leftnext)>10)
                    {
                        ctrl_text += " DOWN";
                    }
                    if(z_leftnext.y<z_leftprev.y&&norm(z_leftprev-z_leftnext)>10)
                    {
                        ctrl_text += " UP";
                    }
                    if(norm(z_leftprev-z_leftnext)<10)
                    {
                        ctrl_text = "STABLE";
                    }
                    
                    z_leftprev.x = z_leftnext.x;
                    z_leftprev.y = z_leftnext.y;
                }
                
                for(int i=0;i<centres1.rows;i++)
                {
                    Point2f p,q;
                    p.x = centres1.at<double>(i,0)-40;
                    p.y = centres1.at<double>(i,1)-40;
                    
                    q.x = centres1.at<double>(i,0)+40;
                    q.y = centres1.at<double>(i,1)+40;
                    rectangle(left_mask1, p, q, CV_RGB(255, 255, 0));
                    
                }
            
                for(int i=0;i<kpts1.size();i++)
                {
                    CvScalar idx1 = GiveColor(labels1.at<int>(i,0));
                    CvScalar idx2 = GiveColor(labels2.at<int>(i,0));
                    circle(left_mask1, kpts1[i], 3, idx1,-1);
                    circle(left_mask2, kpts1[i], 3, idx2,-1);
                }
                
            putText(left_mask1, ctrl_text, Point2f(10, 340 - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, 3);
                
            imshow("left", left_mask1);
            moveWindow("left", 0, 0);
            imshow("leftv", left_mask2);
            moveWindow("leftv", 0, 410);
            //k = waitKey(35);
        }
            if(frame_num>0)
            {
                if(kpts1.size() >= FEATURE_THRESHOLD)
                {
                    prevleftpts.assign(kpts1.begin(), kpts1.end()); // retain previously tracked keypoints
                    
                }else //if(nextleftpts.size()<=MIN_FEATURES)
                {
                    goodFeaturesToTrack(g_greyscale_image_left, prevleftpts, MAX_FEATURES, QUALITY_EIGVALUE, MIN_DISTANCE);
                }
            }
        
            
            g_greyscale_image_left.copyTo(prevleftimg);
            
            
            
            vector<Point2f> nextrightpts,kpts2,vkpts2;
            if(prevrightpts.size()<MIN_FEATURES)
            {
                //control input
                ctrl_text = "STOP MOVING";
            }
            else if(frame_num>0)
            {
                vector<uchar> status;
                vector<float> error;
                vector<Mat> nextrightpyr,prevrightpyr;
                buildOpticalFlowPyramid(prevrightimg, prevrightpyr, Size(7,7), 4);
                buildOpticalFlowPyramid(g_greyscale_image_right, nextrightpyr, Size(7,7), 4);
                CvTermCriteria optical_flow_termination_criteria
                = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.1 );
                calcOpticalFlowPyrLK(prevrightpyr, nextrightpyr, prevrightpts, nextrightpts, status, error,Size(7,7),4,optical_flow_termination_criteria,OPTFLOW_LK_GET_MIN_EIGENVALS);
                
                Mat right_mask1(g_greyscale_image_right.size(),CV_8UC3);
                Mat right_mask2(g_greyscale_image_right.size(),CV_8UC3);
                cvtColor(g_greyscale_image_right, right_mask1, CV_GRAY2BGR);
                cvtColor(g_greyscale_image_right, right_mask2, CV_GRAY2BGR);
                
                for(size_t i=0;i<nextrightpts.size();i++)
                {
                    if(nextrightpts[i].x>=0 && nextrightpts[i].x<WIDTH && nextrightpts[i].y>=0 && nextrightpts[i].y<=HEIGHT)
                    {
                        
                        //
                        CvScalar line_color; line_color = CV_RGB(255,0,0);
                        Point2f p,q;
                        p.x = (double) prevrightpts[i].x;
                        p.y = (double) prevrightpts[i].y;
                        q.x = (double) nextrightpts[i].x;
                        q.y = (double) nextrightpts[i].y;
                        
                        double d = sqrt(pow(p.x-q.x,2)+pow(p.y-q.y,2));
                        
                        if(d<20)
                        {
                            //circle(right_mask, q, 4, Scalar(0,255,0),-1);
                            arrowedLine(right_mask1, p, q, Scalar(255,100,100) , 1, CV_AA, 0, 0.4);
                            arrowedLine(right_mask2, p, q, Scalar(255,100,100) , 1, CV_AA, 0, 0.4);
                            
                        }
                        
                        vkpts2.push_back((q-p)*getTickFrequency()/(t-t1));
                        kpts2.push_back(nextrightpts[i]);
                        
                    }
                }

                ctrl_text = "GO";
                if(kpts2.size()!=0)
                {
                    kmeans(kpts2, clusterNum, labels1, TC, attempts,  KMEANS_PP_CENTERS, centres1);
                    kmeans(vkpts2, clusterNum, labels2, TC, attempts,  KMEANS_PP_CENTERS, centres2);
                    
                    reduce(centres1, mean_, 0, CV_REDUCE_AVG);
                    z_rightnext.x = mean_.at<double>(0,0);
                    z_rightnext.y = mean_.at<double>(0,1);
                    
                    if(z_rightnext.x>z_rightprev.x&&(double)norm(z_rightprev-z_rightnext)>10)
                    {
                        ctrl_text += " RIGHT";
                    }
                    if(z_rightnext.x<z_rightprev.x&&(double)norm(z_rightprev-z_rightnext)>10)
                    {
                        ctrl_text += " LEFT";
                    }
                    if(z_rightnext.y>z_rightprev.y&&(double)norm(z_rightprev-z_rightnext)>10)
                    {
                        ctrl_text += " DOWN";
                    }
                    if(z_rightnext.y<z_rightprev.y&&(double)norm(z_rightprev-z_rightnext)>10)
                    {
                        ctrl_text += " UP";
                    }
                    if((double)norm(z_rightprev-z_rightnext)<10)
                    {
                        ctrl_text = "STABLE";
                    }
                    
                    z_rightprev.x = z_rightnext.x;
                    z_rightprev.y = z_rightnext.y;

                }
                
                for(int i=0;i<centres1.rows;i++)
                {
                    Point2f p,q;
                    p.x = centres1.at<double>(i,0)-40;
                    p.y = centres1.at<double>(i,1)-40;
                    
                    q.x = centres1.at<double>(i,0)+40;
                    q.y = centres1.at<double>(i,1)+40;
                    rectangle(right_mask1, p, q, CV_RGB(255, 255, 0));
                    
                }
                
                for(int i=0;i<kpts2.size();i++)
                {
                    CvScalar idx1 = GiveColor(labels1.at<int>(i,0));
                    CvScalar idx2 = GiveColor(labels2.at<int>(i,0));
                    circle(right_mask1, kpts2[i], 3, idx1,-1);
                    circle(right_mask2, kpts2[i], 3, idx2,-1);
                }
                
                putText(right_mask1, ctrl_text, Point2f(10, HEIGHT - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, 3);
                imshow("right", right_mask1);
                moveWindow("right", 650, 0);
                imshow("rightv", right_mask2);
                moveWindow("rightv", 650, 410);
                k = waitKey(35);
            }
            if(frame_num>0)
            {
                if(kpts2.size() >= FEATURE_THRESHOLD)
                {
                    prevrightpts.assign(kpts2.begin(), kpts2.end()); // retain previously tracked keypoints
                    
                }else //if(nextleftpts.size()<=MIN_FEATURES)
                {
                    goodFeaturesToTrack(g_greyscale_image_right, prevrightpts, MAX_FEATURES, QUALITY_EIGVALUE, MIN_DISTANCE);
                }
            }
            
            
            g_greyscale_image_right.copyTo(prevrightimg);
            
          Mat disp =  StereoMatching(g_greyscale_image_left, g_greyscale_image_right);
          Mat disp8,_3dimage;
        //reprojectImageTo3D(disp,_3dimage, Q);
            
        normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

            
            imshow("3d",disp8);
            
            frame_num++;
            t1 = (double)getTickCount();
        }
    
        
        if (k == 27)
        {
            break;
        }
  
    }
    
    
    cvReleaseCapture(&capture1);
    cvReleaseCapture(&capture2);
    

    
    return 0;
}