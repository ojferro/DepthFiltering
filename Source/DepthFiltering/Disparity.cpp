//Oswaldo Ferro - Last edit: 15 May 2017
#include <opencv2/stereo/stereo.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

/////////////////////GLOBAL VARIABLES////////////////////////
/////////////////////////////////////////////////////////////
Mat imgL;
Mat imgR;
Mat disp16S;
Mat disp8U;
//////////////////////Trackbar Stuff/////////////////////////
int H_MIN = 3;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 0;
int V_MIN = 43;
int V_MAX = 111;

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void findDisparity() {
	//int ndisparities = 128;
	//int SADWindowSize = 25;
	//Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

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

	sbm->compute(imgL, imgR, disp16S);

	double maxVal, minVal;
	minMaxLoc(disp16S, &minVal, &maxVal);
	printf("Min disp: %f Max value: %f \n", minVal, maxVal);

	disp16S.convertTo(disp8U, CV_8UC1, 255/(maxVal-minVal));

}

void on_trackbar(int, void*) {
	
}

void showTrackbars() {	//Standalone function to set parameters. TODO: Make a function that calculates the parameters based on a min and max DISTANCE from the camera.
						//If function is removed, remove global variables "Trackbar Stuff" above
	String windowName = "Trackbars";
	int const numNames = 6;

	char TrackbarNames[numNames];
	sprintf(TrackbarNames, "H_MIN", H_MIN);
	sprintf(TrackbarNames, "H_MAX", H_MAX);
	sprintf(TrackbarNames, "S_MIN", S_MIN);
	sprintf(TrackbarNames, "S_MAX", S_MAX);
	sprintf(TrackbarNames, "V_MIN", V_MIN);
	sprintf(TrackbarNames, "V_MAX", V_MAX);

	namedWindow(windowName, 0);
	createTrackbar("H_MIN", windowName, &H_MIN, H_MAX, NULL);
	createTrackbar("H_MAX", windowName, &H_MAX, H_MAX, NULL);
	createTrackbar("S_MIN", windowName, &S_MIN, S_MAX, NULL);
	createTrackbar("S_MAX", windowName, &S_MAX, S_MAX, NULL);
	createTrackbar("V_MIN", windowName, &V_MIN, V_MAX, NULL);
	createTrackbar("V_MAX", windowName, &V_MAX, V_MAX, NULL);



}

int main(int argc, char** argv) {

	if (argc == 0) {
		cout << "Not enough arguments" << endl;
		return -1;
	}

	imgL = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	imgR = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	disp16S = Mat(imgL.rows, imgL.cols, CV_16S);
	disp8U = Mat(imgL.rows, imgL.cols, CV_8UC1);

	if (imgL.empty() || imgR.empty()) {
		cout << "Could not load image";
		return -1;
	}

	cout << "Calculating Disparity Map...\n";
	findDisparity();
	
	//////////////////////////////OUTPUT////////////////////////
	/*
	namedWindow("Left Img.", WINDOW_AUTOSIZE);
	imshow("Left Img.", imgL);

	namedWindow("Right Img.", WINDOW_AUTOSIZE);
	imshow("Right Img.", imgR);
	*/
	namedWindow("Disparity Map", WINDOW_AUTOSIZE);
	imshow("Disparity Map", disp8U);
	
	
	//imwrite(argv[3], disp8U);
	
	////////////////////////////////////////////////////////////

	showTrackbars();

	Mat HSVimg = Mat::zeros (disp8U.size(), disp8U.type());
	Mat threshold = Mat::zeros (HSVimg.size(), HSVimg.type());
	Mat masked = Mat::zeros(HSVimg.size(), HSVimg.type());
	applyColorMap(disp8U, HSVimg, COLORMAP_RAINBOW);

	namedWindow("HSV Map", WINDOW_AUTOSIZE);
	namedWindow("Threshold", WINDOW_AUTOSIZE);
	namedWindow("Masked", WINDOW_AUTOSIZE);
	while (true) {
		inRange(HSVimg, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
		imshow("HSV Map", HSVimg);
		imshow("Threshold", threshold);
		imwrite("Thresholded.jpg", threshold);

		HSVimg.copyTo(masked, threshold);
		//bitwise_and(threshold, HSVimg, masked);
		imshow ("Masked", masked);
		waitKey(30);
	}
	


	waitKey(0);
	return 0;
}

