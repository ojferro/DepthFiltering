#include <opencv2/stereo/stereo.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>

#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;


int main(int argc, char** argv) {

	if (argc != 3) {
		cout << "Not enough arguments" << endl;
		return -1;
	}

	Mat imgL = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgR = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	Mat disp16S = Mat(imgL.rows, imgL.cols, CV_16S);
	Mat disp8U = Mat(imgL.rows, imgL.cols, CV_8UC1);

	if (imgL.empty() || imgR.empty()) {
		cout << "Could not load image";
		return -1;
	}

	/////////////////////////////////////////////////////////
	int ndisparities = 128;
	int SADWindowSize = 25;
	//Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	Ptr<StereoSGBM> sbm = StereoSGBM::create(0,    //int minDisparity
		96,     //int numDisparities
		5,      //int SADWindowSize
		600,    //int P1 = 0
		2400,   //int P2 = 0
		20,     //int disp12MaxDiff = 0
		16,     //int preFilterCap = 0
		1,      //int uniquenessRatio = 0
		100,    //int speckleWindowSize = 0
		20,     //int speckleRange = 0
		true);  //bool fullDP = false

	sbm->compute(imgL, imgR, disp16S);

	double maxVal, minVal;
	minMaxLoc(disp16S, &minVal, &maxVal);
	printf("Min disp: %f Max value: %f \n", minVal, maxVal);

	disp16S.convertTo(disp8U, CV_8UC1, 255 / (maxVal - minVal));

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", disp8U);
	//imwrite("disp16S.jpg", disp16S);
	//imwrite("disp8U.jpg", disp8U);


	waitKey(0);
	return 0;

}

