//Oswaldo Ferro - Last edit: 23 May 2017

/*
Non-webcam stuff is working fine
Have not tried webcam stuff
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



using namespace cv;
using namespace std;

/////////////////////GLOBAL VARIABLES//////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
Mat imgL;
Mat imgR;
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

VideoCapture capL;
VideoCapture capR;

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

Size imageSize;

const bool webcam = false;
const bool calibrated = true;

const String INTRINSICS_FILE_PATH = "Data/intrinsics.yml";
const String EXTRINSICS_FILE_PATH = "Data/extrinsics.yml";
//////////////////////Masking Trackbar Stuff///////////////////////////////////////////////
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



///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

int validateSAD(int sad) {
	/*return (SADWindowSize < 5) ? 5
		: (SADWindowSize % 2 == 1) ? SADWindowSize : SADWindowSize + 1;
	*/
	/*if (sad < 5)
		SADWindowSize = 5;
	else if (sad % 2 == 0 && SADWindowSize<255)
		SADWindowSize++;*/

	if (sad < 5)
		return 5;
	else if (sad % 2 == 0 && sad<255)
		return (sad +1);
	return sad;
}

int validateNDisp() {
	return (ndisparities % 16 == 0) ? ndisparities : (round(ndisparities / 16.0) * 16); 

}

void findDisparity(Mat imgL, Mat imgR, Rect roiL, Rect roiR) {
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
	/*cout << SADWindowSize << endl;
	cout << "VALIDATED: " << SADWindowSize << endl;
	cout << SADWindowSize << endl;*/
	sbm->setSmallerBlockSize(validateSAD(SADWindowSizeChange));


	

	sbm->compute(imgL, imgR, disp16S);

	/*double maxVal, minVal;
	minMaxLoc(disp16S, &minVal, &maxVal);
	printf("Min disp: %f Max value: %f \n", minVal, maxVal);
	*/

	disp16S.convertTo(disp8U, CV_8UC1, 255/(ndisparities*16.0));
}

void on_trackbar(int, void*) {
	//ndisparities = (ndisparitiesChange % 16 == 0) ? ndisparitiesChange : (ndisparities/16)*16;
}

void maskingTrackbars() {	//Standalone function to set parameters. TODO: Make a function that calculates the parameters based on a min and max DISTANCE from the camera.
	String windowName = "MaskingTrackbars";
	int const numNames = 6;

	char TrackbarNames[numNames];
	sprintf(TrackbarNames, "H_MIN");
	sprintf(TrackbarNames, "H_MAX");
	sprintf(TrackbarNames, "S_MIN");
	sprintf(TrackbarNames, "S_MAX");
	sprintf(TrackbarNames, "V_MIN");
	sprintf(TrackbarNames, "V_MAX");

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

	/*char TrackbarNames[numNames];
	sprintf(TrackbarNames, "Pre Filter Cap");
	sprintf(TrackbarNames, "Pre Filter Type");
	sprintf(TrackbarNames, "Pre Filter Size");
	sprintf(TrackbarNames, "Min Disparity");
	sprintf(TrackbarNames, "Texture Threshold");
	sprintf(TrackbarNames, "Unniqueness Ratio");
	sprintf(TrackbarNames, "Speckle Window Size");
	sprintf(TrackbarNames, "Speckle Range");
	sprintf(TrackbarNames, "Disp12 Max Diff");

	createTrackbar("Pre Filter Cap", windowName, &preFilterCap, 63, NULL);
	createTrackbar("Pre Filter Type", windowName, &preFilterType, 1, NULL);
	createTrackbar("Pre Filter Size", windowName, &preFilterSize, 256, NULL);
	createTrackbar("Min Disparity", windowName, &minDisparity, 256, NULL);
	createTrackbar("Texture Threshold", windowName, &textureThreshold, 256, NULL);
	createTrackbar("Unniqueness Ratio", windowName, &uniquenessRatio, 256, NULL);
	createTrackbar("Speckle Window Size", windowName, &speckleWindowSize, 256, NULL);
	createTrackbar("Speckle Range", windowName, &speckleRange, 256, NULL);
	createTrackbar("Disp12 Max Diff", windowName, &disp12MaxDiff, 256, NULL);
	*/

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

bool init_cams() {
	capL.open(0);
	capR.open(0);
	if (!capL.isOpened() || !capR.isOpened()) {
		cout << "Not open";
		return false;
	}
	
	capL.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capL.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	
	capR.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capR.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
}


void readMats(){
	FileStorage fsIntr(INTRINSICS_FILE_PATH, FileStorage::READ);
	FileStorage fsExtr(EXTRINSICS_FILE_PATH, FileStorage::READ);

	M1.empty() ? cout<<"empty" : cout<<"not empty";

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

	M1.empty() ? cout << "empty" : cout << "not empty"<<endl;
}


int main(int argc, char** argv) {
	cout << StereoBM::PREFILTER_NORMALIZED_RESPONSE << "  " << StereoBM::PREFILTER_XSOBEL << endl << endl;
	if (!calibrated)	//TODO: Do something if it does not find the extrinsics and intrinsic .yml files 
		//Call calibrator

		if (argc == 0) {
			cout << "Not enough arguments" << endl;
			return -1;
		}

	CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1.0|}{nr||}{help||}{@input|../data/stereo_calib.xml|}{iL|Images/meL-1meter.png|}{iR|Images/meR-1meter.png|}");
	String imgLfn = parser.get<string>("iL");
	String imgRfn = parser.get<string>("iR");

	//cout << imgLfn << "  " << imgRfn;

	//Read in intrinsic and extrinsic matrices from calibration	
	readMats();	//TODO: Error checking if matrices cannot be read

	if (!webcam) {
		imgL = imread(imgLfn, IMREAD_GRAYSCALE);
		imgR = imread(imgRfn, IMREAD_GRAYSCALE);
		//cvtColor(cimgL, imgL, COLOR_BGR2GRAY);
		//cvtColor(cimgR, imgR, COLOR_BGR2GRAY);
		//namedWindow("c", WINDOW_AUTOSIZE);
		//imshow("c", cimgL);
		//imshow("g", imgL);
		//waitKey(30);

		disp16S = Mat(imgL.rows, imgL.cols, CV_16S);
		disp8U = Mat(imgL.rows, imgL.cols, CV_8UC1);

		if (imgL.empty() || imgR.empty()) {
			cout << "Could not load image";
			return -1;
		}
		imageSize = imgL.size();
		cout << "Calculating Disparity Map...\n";
		cout << imgL.empty() << "   " << imgL.size();
	}
	else {
		if (!init_cams())
			return 1;
		
		disp16S = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_16S);
		disp8U = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
		
		capL.read(imgL);
		capR.read(imgR);

		imageSize = Size(FRAME_WIDTH, FRAME_HEIGHT);
	}
	
	//////////////////////////////OUTPUT////////////////////////
	
	//namedWindow("Left Img.", WINDOW_AUTOSIZE);
	//imshow("Left Img.", imgL);

	//namedWindow("Right Img.", WINDOW_AUTOSIZE);
	//imshow("Right Img.", imgR);

	//waitKey(0);
	//return 1;
	
	//namedWindow("Disparity Map", WINDOW_AUTOSIZE);
	//imshow("Disparity Map", disp8U);
	
	
	//imwrite(argv[3], disp8U);
	
	////////////////////////////////////////////////////////////

	//maskingTrackbars();
	Rect roiL, roiR;
	stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q , CALIB_ZERO_DISPARITY, -1, imageSize, &roiL, &roiR);

	Mat rmap[2][2];
	initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat rimgL, rimgR;

	remap(imgL, rimgL, rmap[0][0], rmap[0][1], INTER_LINEAR);
	//cvtColor(rimgL, cimgL, COLOR_GRAY2BGR);
	remap(imgR, rimgR, rmap[1][0], rmap[1][1], INTER_LINEAR);
	//cvtColor(rimgR, cimgR, COLOR_GRAY2BGR);

	//imwrite("rectifiedL.png", rimgL);
	//imwrite("rectifiedR.png", rimgR);

	////////////Displaying Rectified Images side by side (debugging)/////////
	/*Mat H;
	hconcat(rimgL, rimgR, H);

	int distBtwnLines = 20;
	for (int l = 0; l < H.rows; l += distBtwnLines)
		line(H, Point(0, l), Point(H.cols, l), Scalar(0, 0, 255));
	rectangle(H, roiL, Scalar(0, 0, 255), 2, 8, 0);
	rectangle(H, Rect(roiR.x+rimgL.cols, roiR.y,roiR.width, roiR.height) , Scalar(0, 0, 255), 2, 8, 0);
	imshow("Combo", H);
	imwrite("Combo.png", H);
	*//////////////////////////////////////////////////////////////////////////

	Rect newRoiL(roiL.x, roiL.y, 950, 550);	// 1100, 850

	Mat crL = rimgL(newRoiL);
	Mat crR = rimgR(newRoiL);
	imwrite("CroppedL.png", crL);
	imwrite("CroppedR.png", crR);

	//THIS IS THE NEW THING FOR DRAWING EPILINES THAT DOESNT WORK YET!!!
	/*vector<uchar> array;
	if (mat.isContinuous()) {
		array.assign(mat.datastart, mat.dataend);
	}
	else {
		for (int i = 0; i < mat.rows; ++i) {
			array.insert(array.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i) + mat.cols);
		}*/
	

	//imshow("rimgL", rimgL);
	//imshow("rimgR", rimgR);

	/////////////////////////////////////////////

	//disparityTrackbars();

	/*Mat crL_colour, crR_colour;
	applyColorMap(crL, crL_colour, COLORMAP_RAINBOW);
	applyColorMap(crR, crR_colour, COLORMAP_RAINBOW);
	imshow ("COLOUR", crL_colour);*/

	Mat crL_contrast, crR_contrast;
	crL.convertTo(crL_contrast, -1, 2, 0);
	crR.convertTo(crR_contrast, -1, 2, 0);
	//imshow ("Contrast", crL_contrast);

	while (true) {
		//TODO: Grab frame
		findDisparity(crL_contrast, crR_contrast, newRoiL, newRoiL);//Outputs disparity map to disp16S
		imshow("Disp8U", disp8U);
		waitKey(30);
		disp8U.setTo(Scalar(0, 0, 0));
		//imshow("Disp8U", disp8U);
		//waitKey(30);
	}

	imwrite("Disp8U.png", disp8U);
	imshow("Disp16S", disp16S);

	//waitKey(0);

	//Mat threshold = disp8U;
	//Mat maskedL = disp8U;
	//Mat maskedR = disp8U;

	//for debugging purposes only//
	//Mat colour_disp8U;	
	//applyColorMap(disp8U, colour_disp8U, COLORMAP_RAINBOW);
	///////////////////////////////
	//Mat crimgL;
	//Mat crimgR;
	//cvtColor(rimgL, crimgL, CV_GRAY2BGR);
	//imshow("crimgL", crimgL);

	/*do {

		imgL.copyTo(maskedL, threshold);
		imgR.copyTo(maskedR, threshold);
		imshow ("MaskedL", maskedL);
		imshow("MaskedR", maskedR);
		String fn = "MaskedImgLSWAPPED.png";
		imwrite(fn, maskedL);
		waitKey(30);
	} while (false);*/
	
	waitKey(0);
	return 0;
}