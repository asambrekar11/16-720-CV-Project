//
//  dji_lkt.cpp
//  Opencv Test
//
//  Created by Akash Sambrekar on 4/11/16.
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
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
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

static const double pi = 3.14159265358979323846;

/* enc_key */
char *key;
/* req_id for nav closed by app msg */
req_id_t nav_force_close_req_id = {0};
/* std msg from uav */
sdk_std_msg_t recv_sdk_std_msgs = {0};
/* ros launch param */
std::string	serial_name;
int			baud_rate;
int			app_id;
int			app_api_level;
int			app_version;
std::string	app_bundle_id;
std::string	enc_key;
/* activation */
activation_data_t activation_msg = {14,2,1,""};
bool cmd_complete = false;

/* parameter */
#define TAKEOFF			(uint8_t) 4
#define LAND			(uint8_t) 6
#define WIDTH			320
#define HEIGHT			240
#define IMAGE_SIZE		(HEIGHT * WIDTH)
#define VBUS			e_vbus1
#define RETURN_IF_ERR(err_code) { if( err_code ){ release_transfer(); printf( "error code:%d,%s %d\n", err_code, __FILE__, __LINE__ );}}

/* guidance */
int			err_code;
Mat     	g_greyscale_image_left=Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
Mat 		g_greyscale_image_right=Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
int			iter = 0;
DJI_lock    g_lock;
DJI_event   g_event;

// Shi-Tomasi
#define MIN_FEATURES (uint) 15
#define FEATURE_THRESHOLD (uint) 95
#define MAX_FEATURES (uint) 100
#define QUALITY_EIGVALUE (double) 0.01
#define MIN_DISTANCE (double) 1

// Lucas-Kanade
int frame_num = 0;
vector<Point2f> prevleftpts, prevrightpts;
Mat prevleftimg, prevrightimg;
#define OF_MARGIN (double) 0.5 // 2 optical flow values considered different if their difference exceeds this margin
bool l_kpt_regen = true;
bool r_kpt_regen = true;

// control
#define CMD_FLAG 0x4A // control horizontal/vertical velocity in body frame and yaw rate in ground frame
#define FWD (double) 0.5 // constant horizontal velocity
#define TURN (double) 10 // constant yaw rate
#define ALT (double) 0.01 // constant vertical velocity
double l_fwd = FWD; // left image forward control
double l_turn = 0; // left image turn control
double l_alt = 0; // left image altitude control
double r_fwd = FWD; // right image forward control
double r_turn = 0; // right image turn control
double r_alt = 0; // right image altitude control
double turn_prev = 0; // previous yaw for weighted camera observation
#define ctrl_strat 0

/*************************************/

/*
 * table of sdk req data handler
 */
int16_t sdk_std_msgs_handler(uint8_t cmd_id,uint8_t* pbuf,uint16_t len,req_id_t req_id);
int16_t	nav_force_close_handler(uint8_t cmd_id,uint8_t* pbuf,uint16_t len,req_id_t req_id);
/* cmd id table */
cmd_handler_table_t cmd_handler_tab[] =
{
    {0x00,sdk_std_msgs_handler				},
    {0x01,nav_force_close_handler			},
    {ERR_INDEX,NULL							}
};
/* cmd set table */
set_handler_table_t set_handler_tab[] =
{
    {0x02,cmd_handler_tab					},
    {ERR_INDEX,NULL							}
};

/*
 * sdk_req_data_callback
 */
int16_t nav_force_close_handler(uint8_t cmd_id,uint8_t* pbuf,uint16_t len,req_id_t req_id)
{
    if(len != sizeof(uint8_t))
        return -1;
    uint8_t msg;
    memcpy(&msg, pbuf, sizeof(msg));
    /* test session ack */
    nav_force_close_req_id.sequence_number = req_id.sequence_number;
    nav_force_close_req_id.session_id      = req_id.session_id;
    nav_force_close_req_id.reserve	       = 1;
    
    printf("WARNING nav close by app %d !!!!!!!!!!!!!! \n", msg);
    return 0;
    
}

#define _recv_std_msgs(_flag, _enable, _data, _buf, _datalen) \
if( (_flag & _enable))\
{\
memcpy((uint8_t *)&(_data),(uint8_t *)(_buf)+(_datalen), sizeof(_data));\
_datalen += sizeof(_data);\
}

int16_t sdk_std_msgs_handler(uint8_t cmd_id,uint8_t* pbuf,uint16_t len,req_id_t req_id)
{
    uint16_t *msg_enable_flag = (uint16_t *)pbuf;
    uint16_t data_len = MSG_ENABLE_FLAG_LEN;
    
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_TIME	, recv_sdk_std_msgs.time_stamp			, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_Q		, recv_sdk_std_msgs.q				, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_A		, recv_sdk_std_msgs.a				, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_V		, recv_sdk_std_msgs.v				, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_W		, recv_sdk_std_msgs.w				, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_POS	, recv_sdk_std_msgs.pos				, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_MAG	, recv_sdk_std_msgs.mag				, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_RC		, recv_sdk_std_msgs.rc				, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_GIMBAL	, recv_sdk_std_msgs.gimbal			, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_STATUS	, recv_sdk_std_msgs.status			, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_BATTERY	, recv_sdk_std_msgs.battery_remaining_capacity	, pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_DEVICE	, recv_sdk_std_msgs.ctrl_device			, pbuf, data_len);
    
    return 0;
}

/* test cmd agency */
uint8_t test_cmd_send_flag = 1;
uint8_t test_cmd_is_resend = 0;
void cmd_callback_test_fun(uint16_t *ack)
{
    char result[6][50]={{"REQ_TIME_OUT"},{"REQ_REFUSE"},{"CMD_RECIEVE"},{"STATUS_CMD_EXECUTING"},{"STATUS_CMD_EXE_FAIL"},{"STATUS_CMD_EXE_SUCCESS"}};
    uint16_t recv_ack = *ack;
    printf("[DEBUG] recv_ack %#x \n", recv_ack);
    printf("[TEST_CMD] Cmd result: %s \n", *(result+recv_ack));
    test_cmd_send_flag = 1;
    if(recv_ack != STATUS_CMD_EXE_SUCCESS)
    {
        test_cmd_is_resend = 1;
    }
    
    /* for debug */
    if(recv_ack != STATUS_CMD_EXE_SUCCESS)
    {
        test_cmd_send_flag  = 0;
        printf("[ERROR] APP LAYER NOT STATUS_CMD_EXE_SUCCESS !!!!!!!!!!!!!!!!!!\n");
    }
    cmd_complete = true;
    printf("Completed Maneuver...\n");
}

/* test activation */
void test_activation_ack_cmd_callback(ProHeader *header)
{
    uint16_t ack_data;
    printf("Sdk_ack_cmd0_callback,sequence_number=%d,session_id=%d,data_len=%d\n", header->sequence_number, header->session_id, header->length - EXC_DATA_SIZE);
    memcpy((uint8_t *)&ack_data,(uint8_t *)&header->magic, (header->length - EXC_DATA_SIZE));
    
    if( is_sys_error(ack_data))
    {
        printf("[DEBUG] SDK_SYS_ERROR!!! \n");
    }
    else
    {
        char result[][50]={{"ACTIVATION_SUCCESS"},{"PARAM_ERROR"},{"DATA_ENC_ERROR"},{"NEW_DEVICE_TRY_AGAIN"},{"DJI_APP_TIMEOUT"},{" DJI_APP_NO_INTERNET"},{"SERVER_REFUSED"},{"LEVEL_ERROR"}};
        printf("[ACTIVATION] Activation result: %s \n", *(result+ack_data));
        if(ack_data == 0)
        {
            Pro_Config_Comm_Encrypt_Key(key);
            printf("[ACTIVATION] set key %s\n",key);
        }
    }
    cmd_complete = true;
    printf("Completed Activation...\n");
}

void test_activation(void)
{
    App_Send_Data( 2, 0, MY_ACTIVATION_SET, API_USER_ACTIVATION,(uint8_t*)&activation_msg,sizeof(activation_msg), test_activation_ack_cmd_callback, 1000, 1);
    printf("[ACTIVATION] send acticition msg: %d %d %d %d \n", activation_msg.app_id, activation_msg.app_api_level, activation_msg.app_ver ,activation_msg.app_bundle_id[0]);
}

void sdk_ack_nav_open_close_callback(ProHeader *header)
{
    uint16_t ack_data;
    printf("call %s\n",__func__);
    printf("Recv ACK,sequence_number=%d,session_id=%d,data_len=%d\n", header->sequence_number, header->session_id, header->length - EXC_DATA_SIZE);
    memcpy((uint8_t *)&ack_data,(uint8_t *)&header->magic, (header->length - EXC_DATA_SIZE));
    
    if( is_sys_error(ack_data))
    {
        printf("[DEBUG] SDK_SYS_ERROR!!! \n");
    }
    cmd_complete = true;
    printf("Completed API Control...\n");
}

// onboard monitor command set
void monitor()
{
    uint8_t pbuf;
    uint16_t *msg_enable_flag = (uint16_t *)(&pbuf);
    uint16_t data_len = MSG_ENABLE_FLAG_LEN;
    
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_TIME	, recv_sdk_std_msgs.time_stamp			, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_Q		, recv_sdk_std_msgs.q				, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_A		, recv_sdk_std_msgs.a				, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_V		, recv_sdk_std_msgs.v				, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_W		, recv_sdk_std_msgs.w				, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_POS	, recv_sdk_std_msgs.pos				, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_MAG	, recv_sdk_std_msgs.mag				, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_RC		, recv_sdk_std_msgs.rc				, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_GIMBAL	, recv_sdk_std_msgs.gimbal			, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_STATUS	, recv_sdk_std_msgs.status			, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_BATTERY	, recv_sdk_std_msgs.battery_remaining_capacity	, &pbuf, data_len);
    _recv_std_msgs( *msg_enable_flag, ENABLE_MSG_DEVICE	, recv_sdk_std_msgs.ctrl_device			, &pbuf, data_len);
}

// maneuvering
void maneuver(int ctrl_flag, double roll_or_x, double pitch_or_y, double thr_z, double yaw)
{
    api_ctrl_without_sensor_data_t motion = {0}; // initialize motion commands
    motion.ctrl_flag = ctrl_flag;
    motion.roll_or_x = roll_or_x;
    motion.pitch_or_y = pitch_or_y;
    motion.thr_z = thr_z;
    motion.yaw = yaw;
    while(true)
    {
        if(cmd_complete)
        {
            cmd_complete = false;
            App_Send_Data(0, 0, MY_CTRL_CMD_SET, API_CTRL_REQUEST, (uint8_t*)&motion, sizeof(motion), NULL, 0, 0); // send command
            cmd_complete = true;
            break;
        }
    }
}

int guidance_callback(int data_type, int data_len, char *content)
{
    g_lock.enter();
    if(e_image == data_type && NULL!=content)
    {
        printf("Success..\n");
        image_data image;
        memcpy((char*)&image,content,sizeof(image));
        printf("frame index:%d, time stamp:%d\n",image.frame_index,image.time_stamp);
        
        for(int d=0;d<CAMERA_PAIR_NUM;d++)
        {
            
            if(image.m_greyscale_image_left[d])
            {
                memcpy(g_greyscale_image_left.data, image.m_greyscale_image_left[d],IMAGE_SIZE);
                
                vector<Point2f> nextleftpts,kpts;
                if(prevleftpts.size()<MIN_FEATURES)
                {
                    //control input
                }
                else if(frame_num>0)
                {
                    //vector<CvPoint2D32f> nextleftpts;
                    vector<uchar> status;
                    vector<float> error;
                    vector<Mat> nextleftpyr,prevleftpyr;
                    buildOpticalFlowPyramid(prevleftimg, prevleftpyr, Size(7,7), 4);
                    buildOpticalFlowPyramid(g_greyscale_image_left, nextleftpyr, Size(7,7), 4);
                    CvTermCriteria optical_flow_termination_criteria
                    = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.1 );
                    calcOpticalFlowPyrLK(prevleftpyr, nextleftpyr, prevleftpts, nextleftpts, status, error,Size(7,7),4,optical_flow_termination_criteria,OPTFLOW_LK_GET_MIN_EIGENVALS);
                    
                    //drawing featues
                    
                    Mat left_mask(g_greyscale_image_left.size(),CV_8UC3);
                    cvtColor(g_greyscale_image_left, left_mask, CV_GRAY2BGR);
                    
                    for(size_t i=0;i<nextleftpts.size();i++)
                    {
                        if(nextleftpts[i].x>=0 && nextleftpts[i].x<=WIDTH && nextleftpts[i].y>=0 && nextleftpts[i].y<=HEIGHT)
                        {
                            CvScalar line_color; line_color = CV_RGB(255,0,0);
                            //int line_thickness = 1;
                            CvPoint p,q;
                            p.x = (int) prevleftpts[i].x;
                            p.y = (int) prevleftpts[i].y;
                            q.x = (int) nextleftpts[i].x;
                            q.y = (int) nextleftpts[i].y;
//                            double angle; angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
//                            double hypotenuse; hypotenuse = sqrt( pow((p.y - q.y),2) + pow((p.x - q.x),2) );
//                            
//                            //scale the line by a factor of 3
//                            q.x = (int) (p.x - 2 * hypotenuse * cos(angle));
//                            q.y = (int) (p.y - 2 * hypotenuse * sin(angle));
                            
                            arrowedLine(left_mask, p, q, line_color);
                            //cvLine( &left_mask, p, q, line_color, line_thickness, CV_AA, 0 );
                           
//                            p.x = (int) (q.x + 9 * cos(angle + pi / 4));
//                            p.y = (int) (q.y + 9 * sin(angle + pi / 4));
//                            cvLine( &prevleftimg, p, q, line_color, line_thickness, CV_AA, 0 );
//                            p.x = (int) (q.x + 9 * cos(angle - pi / 4));
//                            p.y = (int) (q.y + 9 * sin(angle - pi / 4));
//                            cvLine( &prevleftimg, p, q, line_color, line_thickness, CV_AA, 0 );
                            
                            circle(left_mask, nextleftpts[i], 2, Scalar(0,255,0),-1);
                            
                            kpts.push_back(nextleftpts[i]);
                        }
                    }
                    
                    imshow("left", left_mask);
                    moveWindow("left", 0, 0);
                    
                    }
                
                
                
                if(frame_num>0)
                {
                    if(nextleftpts.size()<=MIN_FEATURES)
                    {
                        goodFeaturesToTrack(g_greyscale_image_left, prevleftpts, MAX_FEATURES, QUALITY_EIGVALUE, MIN_DISTANCE);
                    }
                    else
                    {
                        prevleftpts.assign(kpts.begin(), kpts.end());
                    }
                }
                
                g_greyscale_image_left.copyTo(prevleftimg);
                
            }
            
            if(image.m_greyscale_image_right[d])
            {
                memcpy(g_greyscale_image_right.data, image.m_greyscale_image_right[d],IMAGE_SIZE);
                
                vector<Point2f> nextrightpts,kpts;
                if(prevrightpts.size()<MIN_FEATURES)
                {
                    //control input
                }
                else if(frame_num>0)
                {
                    //vector<CvPoint2D32f> nextleftpts;
                    vector<uchar> status;
                    vector<float> error;
                    vector<Mat> nextrightpyr,prevrightpyr;
                    buildOpticalFlowPyramid(prevrightimg, prevrightpyr, Size(7,7), 4);
                    buildOpticalFlowPyramid(g_greyscale_image_right, nextrightpyr, Size(7,7), 4);
                    CvTermCriteria optical_flow_termination_criteria
                    = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.1 );
                    calcOpticalFlowPyrLK(prevrightpyr, nextrightpyr, prevrightpts, nextrightpts, status, error,Size(7,7),4,optical_flow_termination_criteria,OPTFLOW_LK_GET_MIN_EIGENVALS);
                    
                    //drawing featues
                    
                    Mat right_mask(g_greyscale_image_right.size(),CV_8UC3);
                    cvtColor(g_greyscale_image_right, right_mask, CV_GRAY2BGR);
                    
                    for(size_t i=0;i<nextrightpts.size();i++)
                    {
                        if(nextrightpts[i].x>=0 && nextrightpts[i].x<=WIDTH && nextrightpts[i].y>=0 && nextrightpts[i].y<=HEIGHT)
                        {
                            CvScalar line_color; line_color = CV_RGB(255,0,0);
                            //int line_thickness = 1;
                            CvPoint p,q;
                            p.x = (int) prevrightpts[i].x;
                            p.y = (int) prevrightpts[i].y;
                            q.x = (int) nextrightpts[i].x;
                            q.y = (int) nextrightpts[i].y;
//                            double angle; angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
//                            double hypotenuse; hypotenuse = sqrt( pow((p.y - q.y),2) + pow((p.x - q.x),2) );
//                            
////                            //scale the line by a factor of 3
//                            q.x = (int) (p.x - 2 * hypotenuse * cos(angle));
//                            q.y = (int) (p.y - 2 * hypotenuse * sin(angle));
//
//                            cvLine( &prevrightimg, p, q, line_color, line_thickness, CV_AA, 0 );
//                            
//                            p.x = (int) (q.x + 9 * cos(angle + pi / 4));
//                            p.y = (int) (q.y + 9 * sin(angle + pi / 4));
//                            cvLine( &prevrightimg, p, q, line_color, line_thickness, CV_AA, 0 );
//                            p.x = (int) (q.x + 9 * cos(angle - pi / 4));
//                            p.y = (int) (q.y + 9 * sin(angle - pi / 4));
//                            cvLine( &prevrightimg, p, q, line_color, line_thickness, CV_AA, 0 );
//
                            arrowedLine(right_mask, p, q, line_color);
                            circle(right_mask, nextrightpts[i], 2, Scalar(0,255,0),-1);
                            
                            kpts.push_back(nextrightpts[i]);
                        }
                    }
                    
                    imshow("right", right_mask);
                    moveWindow("right", 500, 0);
                    
                }
                
                
                
                if(frame_num>0)
                {
                    if(nextrightpts.size()<=MIN_FEATURES)
                    {
                        goodFeaturesToTrack(g_greyscale_image_right, prevrightpts, MAX_FEATURES, QUALITY_EIGVALUE, MIN_DISTANCE);
                    }
                    else
                    {
                        prevrightpts.assign(kpts.begin(), kpts.end());
                    }
                }
                
                g_greyscale_image_right.copyTo(prevrightimg);
                frame_num++;
            }
            
            
            //Stero matching
            
//            Mat g_l,g_r;
//            
//            cvtColor(g_greyscale_image_left, g_l, CV_BGR2GRAY);
//            cvtColor(g_greyscale_image_right,g_r, CV_BGR2GRAY);
            
            Mat imgDisparity16S = Mat( g_greyscale_image_left.rows, g_greyscale_image_left.cols, CV_16S );
            Mat imgDisparity8U = Mat( g_greyscale_image_left.rows, g_greyscale_image_left.cols, CV_8UC1 );
            
            int ndisparities = 16*4;
            int SADWindowSize = 7;
            
            Ptr<StereoBM> sbm = StereoBM::create( ndisparities, SADWindowSize );
            
            sbm->compute( g_greyscale_image_left, g_greyscale_image_right, imgDisparity16S );
            
            double minVal; double maxVal;
            
            minMaxLoc( imgDisparity16S, &minVal, &maxVal );
            
            printf("Min disp: %f Max value: %f \n", minVal, maxVal);
            
            imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
            
            namedWindow( "windowDisparity", WINDOW_AUTOSIZE );
            imshow( "windowDisparity", imgDisparity8U );
            moveWindow("windowDisparity", 500, 500);

            
        }
        waitKey(1);
        
    }
    
    g_lock.leave();
    g_event.set_DJI_event();
    
    
    return 0;
}




// opening and closing api
void nav_open_close(uint8_t open_close, char *task)
{
    uint8_t send_data = open_close;
    while(true)
    {
        if(cmd_complete)
        {
            printf("\n%s\n", task);
            cmd_complete = false;
            App_Send_Data(1, 1, MY_CTRL_CMD_SET, API_OPEN_SERIAL, (uint8_t *)&send_data, sizeof(send_data), sdk_ack_nav_open_close_callback,  1000, 0); // send command
            break;
        }
    }
}

// takeoff and landing
void take_off_land(uint8_t send_data, char *task)
{
    while(true)
    {
        if(cmd_complete)
        {
            printf("\n%s\n", task);
            cmd_complete = false; // reset cmd_complete
            App_Complex_Send_Cmd(send_data, cmd_callback_test_fun); // send command
            break;
        }
    }
}

// run mission
void run()
{
    /* Takeoff */
    test_activation(); // activate
    nav_open_close(1, (char *)"Opening API Control..."); // open api
    take_off_land(TAKEOFF, (char *)"Taking off..."); // take off
    
    /* Maneuvering and avoiding obstacles */
    usleep(15000000); // pause 15 seconds
    printf("\nBeginning Obstacle Avoidance...\n");
    err_code = start_transfer(); // start guidance data collection
    RETURN_IF_ERR( err_code );
    
    /* Landing */
    getchar(); // press key to begin autonomous mode
    printf("\nEnding Obstacle Avoidance...\n");
    err_code = stop_transfer(); // stop guidance
    RETURN_IF_ERR( err_code );
    monitor();
    while(recv_sdk_std_msgs.pos.height > 0) // check if drone reached ground level yet
    {
        take_off_land(LAND, (char *)"Landing..."); // land
        monitor();
    }
    nav_open_close(0, (char *)"Closing API Control..."); // close api
}

/*
 * main_function
 */
int main (int argc, char** argv)
{
    /* Onboard */
    
    printf("Test SDK Protocol demo\n");
    
    serial_name = std::string("/dev/ttyUSB0");
    baud_rate = 230400;
    app_id = 1010572;
    app_api_level = 2;
    app_version = 1;
    app_bundle_id = std::string("12345678901234567890123456789012");
    enc_key = std::string("ca5aed46d675076dd100ec73a8d3b8d3dbeea66392c77af62ac65cf9b5be8520");
    
    activation_msg.app_id 		= (uint32_t)app_id;
    activation_msg.app_api_level 	= (uint32_t)app_api_level;
    activation_msg.app_ver		= (uint32_t)app_version;
    memcpy(activation_msg.app_bundle_id, app_bundle_id.c_str(), 32);
    
    key = (char*)enc_key.c_str();
    
    printf("[INIT] SET serial_port	: %s \n", serial_name.c_str());
    printf("[INIT] SET baud_rate	: %d \n", baud_rate);
    printf("[INIT] ACTIVATION INFO	: \n");
    printf("[INIT] 	  app_id     	  %d \n", activation_msg.app_id);
    printf("[INIT]    app_api_level	  %d \n", activation_msg.app_api_level);
    printf("[INIT]    app_version     %d \n", activation_msg.app_ver);
    printf("[INIT]    app_bundle_id	  %s \n", activation_msg.app_bundle_id);
    printf("[INIT]    enc_key	  %s \n", key);
    
    /* open serial port */
    Pro_Hw_Setup((char *)serial_name.c_str(),baud_rate);
    Pro_Link_Setup();
    App_Recv_Set_Hook(App_Recv_Req_Data);
    App_Set_Table(set_handler_tab, cmd_handler_tab);
    
    CmdStartThread();
    
    Pro_Config_Comm_Encrypt_Key(key);
    
    /* Guidance */
    
    reset_config();
    err_code = init_transfer();
    RETURN_IF_ERR( err_code );
    
    err_code = select_greyscale_image( VBUS, true );
    RETURN_IF_ERR( err_code );
    err_code = select_greyscale_image( VBUS, false );
    RETURN_IF_ERR( err_code );
    
    err_code = set_sdk_event_handler( guidance_callback ); // set guidance callback
    RETURN_IF_ERR( err_code );
    
    /* Mission */
    
    printf("\nRunning Mission...\n");
    run();
    
    //make sure the ack packet from GUIDANCE is received
    sleep( 1000000 );
    err_code = release_transfer();
    RETURN_IF_ERR( err_code );
    
    return 0;
}