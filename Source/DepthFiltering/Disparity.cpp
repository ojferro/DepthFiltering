//Oswaldo Ferro - Last edit: 02 June 2017

/*
Starting to implement runtime disparity


TODO TUESDAY JUNE 6:
	Implement masking of live colour-feed
	Implement timer in C++ to determine FPS
	Start Optimizing code (minimize needless function calls)
	Research superpixels
*/


#include <opencv2/stereo/stereo.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>

//#include "ICamera.h"
#include "point_grey_cam.h"
#include "point_grey_sim.h"



using namespace cv;
using namespace std;

#define POINT_GREY_FOV 42.5
#define POINT_GREY_FPS 5.0	//USED TO BE 10.0
#define POINT_GREY_EXPOSURE 16
#define POINT_GREY_GAIN 10.0
#define POINT_GREY_WHITE_BALANCE 700, 880

/////////////////////GLOBAL VARIABLES//////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
Mat imgL;
Mat imgR;
uchar* rawL;
uchar* rawR;
Mat cimgL;
Mat cimgR;
Mat disp16S;
Mat disp8U;
Mat M1;
Mat M2;
Mat D1;
Mat D2;
Mat R;
Mat T;
Mat R1;
Mat R2;
Mat P1;
Mat P2;
Mat Q;

Mat imgBayerL;
Mat imgBayerR;

Mat rmap[2][2];
Mat rimgL, rimgR;
Rect roiL, roiR, newRoi;
Size roiDimensions(1100, 850);

Mat crL;
Mat crR;

Mat maskedL;
Mat maskedR;
Mat thresh;
Mat threshTemp;

int CLOSE_THRESH = 255;
int FAR_THRESH = 61;	//61 works the best

//VideoCapture capL;
//VideoCapture capR;


Size imageSize;

const bool webcam = true;
const bool calibrated = true;
const bool postProcess = true;
const bool preProcess = true;

const String INTRINSICS_FILE_PATH = "Data/intrinsics.yml";
const String EXTRINSICS_FILE_PATH = "Data/extrinsics.yml";
//////////////////////Masking Trackbar Variables///////////////////////////////////////////
int H_MIN = 3;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 43;
int V_MAX = 111;
//////////////////////Disparity Trackbar Variables/////////////////////////////////////////
int ndisparities = 128;
int SADWindowSize = 9;
int SADWindowSizeChange = SADWindowSize;
int minDisparity = 68;	//the negative is accounteed for later on (i.e. for minDisp of -60, initialize to 60)
int preFilterCap = 30;
int preFilterType = 1;
int preFilterSize = 40;
int textureThreshold = 168;
int uniquenessRatio = 1;
int speckleWindowSize = 12;
int speckleRange = 256;
int disp12MaxDiff = 0;

////////////////////////////////Point Grey Cameras/////////////////////////////////////////
point_grey_camera_manager * GigeManager = 0;
ICamera * PointGreyCam = 0;
ICamera * PointGreyCam2 = 0;

int FRAME_WIDTH;
int FRAME_HEIGHT;

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

int init_PointGrey()
{
	GigeManager = new point_grey_camera_manager(true);
	int num_cameras = GigeManager->get_num_cameras();

	if (num_cameras > 0)
	{
		printf("%d Cameras found. Opening the first one.\n", num_cameras);
		PointGreyCam = GigeManager->get_cam_from_index(0);

		PointGreyCam->set_auto_exposure(false);
		PointGreyCam->set_frame_rate(POINT_GREY_FPS);
		PointGreyCam->set_exposure_ms(POINT_GREY_EXPOSURE);
		PointGreyCam->set_gain_db(POINT_GREY_GAIN);
		PointGreyCam->set_white_balance(POINT_GREY_WHITE_BALANCE);
		FRAME_WIDTH = PointGreyCam->get_width();
		FRAME_HEIGHT = PointGreyCam->get_height();
	}
	else
	{
		printf("Error. %d Cameras found. Exiting.\n", num_cameras);
		waitKey(2000);
		exit(1);
	}

	if (num_cameras > 1)
	{
		printf("%d Cameras found. Opening the second one.\n", num_cameras);
		PointGreyCam2 = GigeManager->get_cam_from_index(1);

		PointGreyCam2->set_auto_exposure(false);
		PointGreyCam2->set_frame_rate(POINT_GREY_FPS);
		PointGreyCam2->set_exposure_ms(POINT_GREY_EXPOSURE);
		PointGreyCam2->set_gain_db(POINT_GREY_GAIN);
		PointGreyCam2->set_white_balance(POINT_GREY_WHITE_BALANCE);
	}
	return 0;
}

int validateSAD(int sad) {
	if (sad < 5)
		return 5;
	else if (sad % 2 == 0 && sad<255)
		return (sad +1);
	return sad;
}

int validateNDisp() {
	return (ndisparities % 16 == 0) ? ndisparities : (round(ndisparities / 16.0) * 16); 

}

void findDisparity(Mat L, Mat R, Rect roiL, Rect roiR) {
	//int ndisparities = 128; //16
	//int SADWindowSize = 9; //9
	//int minDisparity = 0;
	
	///////////////////////STEREO SGBM////////////////////////////////
	//Ptr<StereoSGBM> sbm = StereoSGBM::create(
	//	minDisparity,    //int minDisparity
	//	ndisparities,     //int numDisparities
	//	SADWindowSize,    //int SADWindowSize ~~ 5
	//	600,    //int P1 = 0  ~~600
	//	2400,   //int P2 = 0	~~2400
	//	20,     //int disp12MaxDiff = 0
	//	16,     //int preFilterCap = 0
	//	5,      //int 0 = uniquenessRatio	~~1
	//	100,    //int speckleWindowSize = 0
	//	20,     //int speckleRange = 0
	//	false);  //bool fullDP = false
	//////////////////////////////////////////////////////////////////

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

	//sbm->setROI1(roiL);
	//sbm->setROI2(roiR);

	sbm->setPreFilterCap(preFilterCap == 0 ? preFilterCap + 1 : preFilterCap);	//must be > 0
	sbm->setPreFilterType(preFilterType);
	//if (preFilterType==0)
	//	sbm->setPreFilterSize(preFilterSize);	//issues
	sbm->setMinDisparity(-minDisparity);
	sbm->setTextureThreshold(textureThreshold);
	sbm->setUniquenessRatio (uniquenessRatio);
	sbm->setSpeckleWindowSize (speckleWindowSize);
	sbm->setSpeckleRange (speckleRange);
	sbm->setDisp12MaxDiff(disp12MaxDiff);
	sbm->setNumDisparities(validateNDisp());
	sbm->setSmallerBlockSize(validateSAD(SADWindowSizeChange));

	sbm->compute(L, R, disp16S);

	/*double maxVal, minVal;
	minMaxLoc(disp16S, &minVal, &maxVal);
	printf("Min disp: %f Max value: %f \n", minVal, maxVal);
	*/

	//normalize(disp16S, disp8U, 0, 255, CV_MINMAX, CV_8U);

	disp16S.convertTo(disp8U, CV_8UC1, 255/(ndisparities*16.0));
}

void on_trackbar(int, void*) {
	//ndisparities = (ndisparitiesChange % 16 == 0) ? ndisparitiesChange : (ndisparities/16)*16;
}

void maskingTrackbars() {	//Standalone function to set parameters. TODO: Make a function that calculates the parameters based on a min and max DISTANCE from the camera.
	String windowName = "MaskingTrackbars";

	namedWindow(windowName, 0);
	createTrackbar("H_MIN", windowName, &H_MIN, H_MAX, NULL);
	createTrackbar("H_MAX", windowName, &H_MAX, H_MAX, NULL);
	createTrackbar("S_MIN", windowName, &S_MIN, S_MAX, NULL);
	createTrackbar("S_MAX", windowName, &S_MAX, S_MAX, NULL);
	createTrackbar("V_MIN", windowName, &V_MIN, V_MAX, NULL);
	createTrackbar("V_MAX", windowName, &V_MAX, V_MAX, NULL);

}

void disparityTrackbars() {
	String windowName = "DisparityTrackbars";
	int const numNames = 9;

	namedWindow(windowName, 0);
	createTrackbar("PFCap", windowName, &preFilterCap, 63, NULL);
	createTrackbar("PFType", windowName, &preFilterType, 1, NULL);
	createTrackbar("PFSize", windowName, &preFilterSize, 256, NULL);
	createTrackbar("Min Disp", windowName, &minDisparity, 256, NULL);
	createTrackbar("Tex Thres", windowName, &textureThreshold, 1024, NULL);
	createTrackbar("UnniqRatio", windowName, &uniquenessRatio, 64, NULL);
	createTrackbar("SpeckWinSz", windowName, &speckleWindowSize, 256, NULL);
	createTrackbar("SpeckRange", windowName, &speckleRange, 256, NULL);
	createTrackbar("DispMDiff", windowName, &disp12MaxDiff, 256, NULL);
	createTrackbar("nDisp", windowName, &ndisparities, 256, NULL);
	createTrackbar("SADWin", windowName, &SADWindowSize, 255, NULL);
}

void threshTrackbars() {
	String windowName = "ThreshTrackbars";

	namedWindow(windowName, 0);
	createTrackbar("CLOSE_THRESH", windowName, &CLOSE_THRESH, 255, NULL);
	createTrackbar("FAR_THRESH", windowName, &FAR_THRESH, 255, NULL);
}

///////////////Initializing (regular) Webcams (not PointGrey)//////////////////
//bool init_cams() {
//	capL.open(0);
//	capR.open(0);
//	if (!capL.isOpened() || !capR.isOpened()) {
//		cout << "Not open";
//		return false;
//	}
//	
//	capL.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
//	capL.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
//	
//	capR.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
//	capR.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
//}


bool readMats(){
	FileStorage fsIntr(INTRINSICS_FILE_PATH, FileStorage::READ);
	FileStorage fsExtr(EXTRINSICS_FILE_PATH, FileStorage::READ);

	//TODO: Error checking if matrices cannot be read
	fsIntr["M1"] >> M1;
	fsIntr["D1"] >> D1;
	fsIntr["M2"] >> M2;
	fsIntr["D2"] >> D2;
	fsExtr["R"] >> R;
	fsExtr["T"] >> T;
	fsExtr["R1"] >> R1;
	fsExtr["R2"] >> R2;
	fsExtr["P1"] >> P1;
	fsExtr["P2"] >> P2;
	fsExtr["Q"] >> Q;

	if (M1.empty() || D1.empty() || M2.empty() || D2.empty() || R.empty() || T.empty() || R1.empty() || R2.empty() || P1.empty() || P2.empty() || Q.empty())
		return false;

	return true;
}

void preProc() {

	crL.convertTo(crL, -1, 2, 0);
	crR.convertTo(crR, -1, 2, 0);
	//imshow("CONTRAST", crL);
}

void postProc (){
	Mat temp, kernel;
	/*erode(thresh, temp, kernel);
	dilate(temp, thresh, kernel);*/
	
	//medianBlur(thresh, thresh, 5);		///Works the best
	//GaussianBlur(thresh, thresh, Size(5, 5), 75, 0, 4);
	//morphologyEx(thresh, thresh, MORPH_GRADIENT, getStructuringElement(MORPH_RECT, Size(7, 7)));
	//floodFill(thresh, Point(FRAME_WIDTH/2, FRAME_HEIGHT/2), Scalar(255));
	//blur(thresh, thresh, Size (5, 5));
	//Mat temp; 
	//bilateralFilter(thresh, temp, 5, 75, 75);
	//imshow("Temp", temp);
	
}


int main(int argc, char** argv) {
	if (!calibrated)	//TODO: Do something if it does not find the extrinsics and intrinsic .yml files 
		//Call calibrator

		if (argc == 0) {
			cout << "Not enough arguments" << endl;
			return -1;
		}

	CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1.0|}{nr||}{help||}{@input|../data/stereo_calib.xml|}{iL|Images/meL-1meter.png|}{iR|Images/meR-1meter.png|}");
	String imgLfn = parser.get<string>("iL");
	String imgRfn = parser.get<string>("iR");

	//Read in intrinsic and extrinsic matrices from calibration	
	if (!readMats())
		return false;

	if (!webcam) {
		imgL = imread(imgLfn, IMREAD_GRAYSCALE);
		imgR = imread(imgRfn, IMREAD_GRAYSCALE);
		//cvtColor(cimgL, imgL, COLOR_BGR2GRAY);
		//cvtColor(cimgR, imgR, COLOR_BGR2GRAY);

		disp16S = Mat(imgL.rows, imgL.cols, CV_16S);
		disp8U = Mat(imgL.rows, imgL.cols, CV_8UC1);

		if (imgL.empty() || imgR.empty()) {
			cout << "Could not load image";
			return -1;
		}
		imageSize = imgL.size();
		cout << "Calculating Disparity Map...\n";
	}
	else {
		if (init_PointGrey()) {
			cout << "Unable to star camera(s)";
			waitKey(0);
			return 1;
		}
		
		disp16S = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_16S);
		disp8U = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);

		//Initial read/demosaic
		rawL = PointGreyCam->get_raw_data();
		imgBayerL = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1, rawL, Mat::AUTO_STEP);
		cvtColor(imgBayerL, imgL, COLOR_BayerRG2GRAY);

		rawR = PointGreyCam2->get_raw_data();
		imgBayerR = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1, rawR, Mat::AUTO_STEP);
		cvtColor(imgBayerR, imgR, COLOR_BayerRG2GRAY);

		imageSize = imgL.size();	//TODO: Error trap if imgL and imgR are not the same size
		cout << "Calculating Disparity Map...\n";
	}
	
	//maskingTrackbars();

	stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &roiL, &roiR);


	initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	disparityTrackbars();
	threshTrackbars();


	//TODO: Find more universal way to determine intersection of both ROIs
	newRoi = Rect(roiR.x, roiR.y, roiDimensions.width, roiDimensions.height);	// 1100, 850..... 950, 550

	//Read, Demosaic, find Disp, Mask
	while (true) {
		//READ
		rawL = PointGreyCam->get_raw_data();
		rawR = PointGreyCam2->get_raw_data();
		if (rawL == NULL||rawR == NULL) {
			cout << "Failed to read raw";
			waitKey(0);
		}

		//DEMOSAIC
		imgBayerL = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1, rawL, Mat::AUTO_STEP);		
		imgBayerR = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1, rawR, Mat::AUTO_STEP);
		cvtColor(imgBayerL, cimgL, COLOR_BayerBG2BGR);
		cvtColor(imgBayerR, cimgR, COLOR_BayerBG2BGR);

		//FIND DISPARITY
		remap(cimgL, cimgL, rmap[0][0], rmap[0][1], INTER_LINEAR);
		remap(cimgR, cimgR, rmap[1][0], rmap[1][1], INTER_LINEAR);

		////////////Displaying Rectified Images side by side (debugging)/////////
		/*Mat H;
		hconcat(rimgL, rimgR, H);

		int distBtwnLines = 20;
		for (int l = 0; l < H.rows; l += distBtwnLines)
			line(H, Point(0, l), Point(H.cols, l), Scalar(0, 0, 255));
		rectangle(H, roiL, Scalar(0, 0, 255), 2, 8, 0);
		rectangle(H, Rect(roiR.x+rimgL.cols, roiR.y,roiR.width, roiR.height) , Scalar(0, 0, 255), 2, 8, 0);
		imshow("Combo", H);*/
		/////////////////////////////////////////////////////////////////////////

		//Cropping to ROI size
		cimgL = cimgL(newRoi);
		cimgR = cimgR(newRoi);

		//Readying imgs to find disparity
		cvtColor(cimgL, crL, COLOR_BGR2GRAY);
		cvtColor(cimgR, crR, COLOR_BGR2GRAY);
		if (preProcess)
			preProc();
		
		findDisparity(crL, crR, newRoi, newRoi);//Outputs disparity map to disp16S
		//imshow("Disp8U", disp8U);

		//thresh = disp8U;
		//threshold(disp8U, threshTemp, CLOSE_THRESH, 0, 4);
		//threshold(threshTemp, thresh, FAR_THRESH, 255, 3);	//4 = Threshold to Zero Inverted
		threshold(disp8U, thresh, FAR_THRESH, 255, THRESH_BINARY);

		////////////Attempts at post processing threshold////////////////
		if (postProcess)
			postProc();
		imshow("Thresh", thresh);

		//MASK
		maskedL = imgL;
		maskedR = imgR;

		cimgL.copyTo(maskedL, thresh);
		cimgR.copyTo(maskedR, thresh);
		imshow("MaskedL", maskedL);
		//imshow("MaskedR", maskedR);

		waitKey(1);
	}

	imwrite("Disp8U.png", disp8U);
	imshow("Disp16S", disp16S);

	//waitKey(0);

	

	//for debugging purposes only//
	//Mat colour_disp8U;	
	//applyColorMap(disp8U, colour_disp8U, COLORMAP_RAINBOW);
	///////////////////////////////
	//Mat crimgL;
	//Mat crimgR;
	//cvtColor(rimgL, crimgL, CV_GRAY2BGR);
	//imshow("crimgL", crimgL);


	//String fn = "MaskedImgLSWAPPED.png";
	//imwrite(fn, maskedL);

	
	waitKey(0);
	return 0;
}