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
//////////////////////Trackbar Stuff///////////////////////////////////////////////////////
int H_MIN = 3;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 43;
int V_MAX = 111;

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void findDisparity(Mat imgL, Mat imgR) {
	//int ndisparities = 128;
	//int SADWindowSize = 25;
	//Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
/*
	Ptr<StereoSGBM> sbm = StereoSGBM::create(
		0,    //int minDisparity
		96,     //int numDisparities
		21,      //int SADWindowSize ~~ 5
		600,    //int P1 = 0
		2400,   //int P2 = 0
		20,     //int disp12MaxDiff = 0
		16,     //int preFilterCap = 0
		1,      //int 0 = uniquenessRatio
		100,    //int speckleWindowSize = 0
		20,     //int speckleRange = 0
		true);  //bool fullDP = false
*/
	int ndisparities = 16; //512
	int SADWindowSize = 9; //21

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	sbm->setPreFilterCap (32);

	sbm->compute(imgL, imgR, disp16S);

	double maxVal, minVal;
	minMaxLoc(disp16S, &minVal, &maxVal);
	printf("Min disp: %f Max value: %f \n", minVal, maxVal);

	disp16S.convertTo(disp8U, CV_8UC1, 255/(maxVal - minVal));

}

void on_trackbar(int, void*) {
	
}

void showTrackbars() {	//Standalone function to set parameters. TODO: Make a function that calculates the parameters based on a min and max DISTANCE from the camera.
	String windowName = "Trackbars";
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

	/*cout << M1<<endl;
	cout << D2<<endl;
	cout << R1 << endl;
	*/
}


int main(int argc, char** argv) {

	if (!calibrated)	//TODO: Do something if it does not find the extrinsics and intrinsic .yml files 
		//Call calibrator

		if (argc == 0) {
			cout << "Not enough arguments" << endl;
			return -1;
		}

	CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1.0|}{nr||}{help||}{@input|../data/stereo_calib.xml|}{iL|Images/L_calibration000.png|}{iR|Images/R_calibration000.png|}");
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

	showTrackbars();
	stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0.0, imageSize, 0, 0);

	Mat rmap[2][2];
	initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	/*Mat canvas;
	double sf;
	int w, h;
	sf = 300. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h * 2, w, CV_8UC3);*/

	Mat rimgL, rimgR;// cimgL, cimgR;
	remap(imgL, rimgL, rmap[0][0], rmap[0][1], INTER_LINEAR);
	cvtColor(rimgL, cimgL, COLOR_GRAY2BGR);
	remap(imgR, rimgR, rmap[1][0], rmap[1][1], INTER_LINEAR);
	cvtColor(rimgR, cimgR, COLOR_GRAY2BGR);

	imshow("rimgL", rimgL);
	imshow("rimgR", rimgR);

	/////////////////////////////////////////////
	findDisparity(rimgL, rimgR);//Outputs disparity map to disp16S
	imshow("Disp8U", disp8U);
	waitKey(0);

	Mat HSVimg = Mat::zeros (disp8U.size(), disp8U.type());
	Mat threshold = HSVimg;
	Mat masked = HSVimg;
	applyColorMap(disp8U, HSVimg, COLORMAP_RAINBOW);

	namedWindow("HSV Map", WINDOW_AUTOSIZE);
	namedWindow("Threshold", WINDOW_AUTOSIZE);
	namedWindow("Masked", WINDOW_AUTOSIZE);
	do {
		inRange(HSVimg, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
		imshow("HSV Map", HSVimg);
		imshow("Threshold", threshold);
		imwrite("Thresholded.jpg", threshold);

		HSVimg.copyTo(masked, threshold);
		imshow ("Masked", masked);
		waitKey(30);
	} while (false);
	
	waitKey(0);
	return 0;
}

