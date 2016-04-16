//
//  LK_Pyramid.cpp
//  Opencv Test
//
//  Created by Akash Sambrekar on 3/29/16.
//  Copyright © 2016 CMU. All rights reserved.
//

/* Onboard */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include "DJI_Pro_Codec.h"
#include "DJI_Pro_Hw.h"
#include "DJI_Pro_Link.h"
#include "DJI_Pro_App.h"

/* Guidance */
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "DJI_guidance.h"
#include "DJI_utility.h"
#include "imagetransfer.h"
#include "usb_transfer.h"
using namespace cv;
using namespace std;

/* additional includes */
#include <string.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <thread>

char key;

int main(void)
{
    cvNamedWindow("Camera_Output", 1);    //Create window
    
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
    while(1){ //Create infinte loop for live streaming
        
        IplImage* frame = cvQueryFrame(capture); //Create image frames from capture
    
        cvShowImage("Camera_Output", frame);   //Show image frames on created window
        key = cvWaitKey(10);     //Capture Keyboard stroke
        if (char(key) == 27){
            break;      //If you hit ESC key loop will break.
        }
    }
    cvReleaseCapture(&capture); //Release capture.
    cvDestroyWindow("Camera_Output"); //Destroy Window

    
    
    return 0;
}
