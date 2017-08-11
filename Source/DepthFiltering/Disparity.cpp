//Oswaldo Ferro - Interaptix

#include <opencv2/stereo/stereo.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/aruco.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/seeds.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "point_grey_cam.h"
#include "point_grey_sim.h"
#include <GL/glew.h>
//#include <GLFW/glfw3.h>
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
#include <Windows.h>

using namespace cv;
using namespace std;

#define POINT_GREY_FOV 42.5
#define POINT_GREY_FPS 60.0
#define POINT_GREY_EXPOSURE 16
#define POINT_GREY_GAIN 10.0
#define POINT_GREY_WHITE_BALANCE 700, 880

#define ESC_KEY 27

/////////////////////GLOBAL VARIABLES//////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
Mat fakeBackgroundImg;
bool useFakeBackground = true;
Mat realBackground;
Mat diff_BG_FG; //differences between the background and the foreground
bool useRealBackground = false;

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
Rect roiL, roiR, newRoi;

Mat crL;
Mat crR;

Mat maskedL;
Mat maskedR;
Mat thresh;
Mat thresh_3D;
Mat threshTemp;

String imgLfn;
String imgRfn;

//3D reprojection
FileStorage PointWriter("point_cloud.yml", FileStorage::WRITE);
Mat pointMat;
int validPtCount = 0;

bool show3D = true;


//For Superpixels/////////
int numSuperpixels = 500;
int numLevels = 4;
int prior = 2;
int histogramBins = 5;
int numIterations = 6;
int maxLabel;
int regionSize = 75;
int ruler = 50;

Mat labels;
Mat superpixelEdges;
Mat labelMask;
Mat superpixelatedImg;    //Debugging only (to show the superpixels on img. Not needed for ultimate purpose of program)
Mat filteredImg;
Mat disp8U_filtered;

Ptr<ximgproc::SuperpixelSLIC> seeds;
//////////////////////////

//For user input//
char key = ' ';
bool paused = false;
////////////////////

//Depth filtering thresholds
int CLOSE_THRESH = 255;
int FAR_THRESH = 72;    //61 works the best
int FAR_AVG_THRESH = 82;

Size imageSize;

const bool webcam = true;
bool postProcess = true;
const bool preProcess = true;
const bool showDebugImgs = true;
const bool saveDebugImgs = false;
const bool realCameras = false;

const String INTRINSICS_FILE_PATH = "Data/intrinsics.yml";
const String EXTRINSICS_FILE_PATH = "Data/extrinsics.yml";
char* PATH_TO_VIDEOS = "Videos/";
int MAT_CONVERSION_CHANNELS = CV_8UC1;

//Saving Debug Imgs
VideoWriter VWdisp8U;
VideoWriter VWsuperpixelatedImg;
VideoWriter VWthresh;

//Point cloud mouse control
float angle = 0.0f;
float currentAngle = 0.0f;
int button = -1;
int xOrigin = -1;

//////////////////////Masking Trackbar Variables///////////////////////////////////////////
int H_MIN = 3;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 43;
int V_MAX = 111;
//////////////////////Disparity Trackbar Variables/////////////////////////////////////////
int ndisparities = 128;//128
int SADWindowSize = 9;
int SADWindowSizeChange = SADWindowSize;
int minDisparity = 68;    //the negative is accounteed for later on (i.e. for minDisp of -60, initialize to 60)
int preFilterCap = 30;
int preFilterType = 1;
int preFilterSize = 40;
int textureThreshold = 168;
int uniquenessRatio = 1;
int speckleWindowSize = 12;
int speckleRange = 256;
int disp12MaxDiff = 0;

//////////////////////PointCloud Trackbar Variables/////////////////////////////////////////

int xPtCloud = 500, yPtCloud = 500, zPtCloud = 500;
int scale = 1;
float scaleDown = 1;

////////////////////////////////Point Grey Cameras/////////////////////////////////////////
point_grey_camera_manager * GigeManager = 0;
ICamera * PointGreyCam = 0;
ICamera * PointGreyCam2 = 0;

int FRAME_WIDTH;
int FRAME_HEIGHT;
///////////////////////////////////////////////////////////////////////////////////////////
Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
///////////////////////////////////////////////////////////////////////////////////////////

void endProgram() {
    VWthresh.release();
    VWsuperpixelatedImg.release();
    VWdisp8U.release();

    delete PointGreyCam;
    delete PointGreyCam2;

    destroyAllWindows();

    printf("Ending Program...");
    waitKey(0);
}


int init_PointGrey()
{
    GigeManager = new point_grey_camera_manager(false);
    int num_cameras = GigeManager->get_num_cameras();

    if (num_cameras > 0)
    {
        printf("%d Cameras found. Opening the first one.\n", num_cameras);
        if (!realCameras) {
            PointGreyCam = GigeManager->get_cam_from_index(0, "Videos/");
            MAT_CONVERSION_CHANNELS = CV_8UC3;
        }
        else {
            PointGreyCam = GigeManager->get_cam_from_index(0);
            MAT_CONVERSION_CHANNELS = CV_8UC1;
        }

        PointGreyCam->set_auto_exposure(false);
        PointGreyCam->set_frame_rate(POINT_GREY_FPS);
        PointGreyCam->set_exposure_ms(POINT_GREY_EXPOSURE);
        PointGreyCam->set_gain_db(POINT_GREY_GAIN);
        PointGreyCam->set_white_balance(POINT_GREY_WHITE_BALANCE);
        FRAME_WIDTH = PointGreyCam->get_width();
        FRAME_HEIGHT = PointGreyCam->get_height();
        PointGreyCam->start_capture();
        Sleep(10);
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nFRAME HEIGHT" << FRAME_HEIGHT << endl;
    }
    else
    {
        printf("Error. %d Cameras found. Exiting.\n", num_cameras);
        waitKey(0);
        exit(1);
    }

    if (num_cameras > 1)
    {
        printf("%d Cameras found. Opening the second one.\n", num_cameras);

        if (!realCameras) {
            PointGreyCam2 = GigeManager->get_cam_from_index(1, "Videos/");
        }
        else {
            PointGreyCam2 = GigeManager->get_cam_from_index(1);
        }

        PointGreyCam2->set_auto_exposure(false);
        PointGreyCam2->set_frame_rate(POINT_GREY_FPS);
        PointGreyCam2->set_exposure_ms(POINT_GREY_EXPOSURE);
        PointGreyCam2->set_gain_db(POINT_GREY_GAIN);
        PointGreyCam2->set_white_balance(POINT_GREY_WHITE_BALANCE);
        PointGreyCam2->start_capture();
        Sleep(10);
    }
    Sleep(100);
    return 0;
}


int validateSAD(int sad) {
    if (sad < 5)
        return 5;
    else if (sad % 2 == 0 && sad < 255)
        return (sad + 1);
    else if (sad > 255)
        return 255;
    return sad;
}

int validateNDisp() {
    return (ndisparities % 16 == 0) ? ndisparities : (round(ndisparities / 16.0) * 16); 

}

void init_sbm(int, void*) {
    sbm->setPreFilterCap(preFilterCap == 0 ? preFilterCap + 1 : preFilterCap);    //must be > 0
    sbm->setPreFilterType(preFilterType);
    //if (preFilterType==0)
    //    sbm->setPreFilterSize(preFilterSize);    //must be odd number
    sbm->setMinDisparity(-minDisparity);
    sbm->setTextureThreshold(textureThreshold);
    sbm->setUniquenessRatio (uniquenessRatio);
    sbm->setSpeckleWindowSize (speckleWindowSize);
    sbm->setSpeckleRange (speckleRange);
    sbm->setDisp12MaxDiff(disp12MaxDiff);
    sbm->setNumDisparities(validateNDisp());
    sbm->setSmallerBlockSize(validateSAD(SADWindowSizeChange));
}

/////////////////////////TRACKBARS//////////////////////////
void maskingTrackbars() {//TODO: Make a function that calculates the parameters based on a min and max DISTANCE from the camera.
    String windowName = "MaskingTrackbars";
    namedWindow(windowName, 0);
    resizeWindow(windowName, 400, 300);

    createTrackbar("H_MIN", windowName, &H_MIN, H_MAX, NULL);
    createTrackbar("H_MAX", windowName, &H_MAX, H_MAX, NULL);
    createTrackbar("S_MIN", windowName, &S_MIN, S_MAX, NULL);
    createTrackbar("S_MAX", windowName, &S_MAX, S_MAX, NULL);
    createTrackbar("V_MIN", windowName, &V_MIN, V_MAX, NULL);
    createTrackbar("V_MAX", windowName, &V_MAX, V_MAX, NULL);

}

void disparityTrackbars() {
    String windowName = "DisparityTrackbars";
    namedWindow(windowName, 0);
    resizeWindow(windowName, 400, 600);

    createTrackbar("PFCap", windowName, &preFilterCap, 63, init_sbm);
    createTrackbar("PFType", windowName, &preFilterType, 1, init_sbm);
    createTrackbar("PFSize", windowName, &preFilterSize, 256, init_sbm);
    createTrackbar("Min Disp", windowName, &minDisparity, 256, init_sbm);
    createTrackbar("Tex Thres", windowName, &textureThreshold, 1024, init_sbm);
    createTrackbar("UnniqRatio", windowName, &uniquenessRatio, 64, init_sbm);
    createTrackbar("SpeckWinSz", windowName, &speckleWindowSize, 256, init_sbm);
    createTrackbar("SpeckRange", windowName, &speckleRange, 256, init_sbm);
    createTrackbar("DispMDiff", windowName, &disp12MaxDiff, 256, init_sbm);
    createTrackbar("nDisp", windowName, &ndisparities, 256, init_sbm);
    createTrackbar("SADWin", windowName, &SADWindowSize, 255, init_sbm);
}

void threshTrackbars() {
    String windowName = "ThreshTrackbars";
    namedWindow(windowName, 0);
    resizeWindow(windowName, 400, 200);
    createTrackbar("CLOSE_THRESH", windowName, &CLOSE_THRESH, 255, NULL);
    createTrackbar("FAR_THRESH", windowName, &FAR_THRESH, 255, NULL);
    createTrackbar("FAR_AVG_THRESH", windowName, &FAR_AVG_THRESH, 255, NULL);
}

void superPixelTrackbars() {
    //int numSuperpixels = 500, numLevels = 4, prior = 2, histogramBins = 5, numIterations = 8, maxLabel;

    String windowName = "superPX Trackbars";

    namedWindow(windowName, 0);
    resizeWindow(windowName, 400, 400);

    createTrackbar("numSuperpx", windowName, &numSuperpixels, 1000, NULL);
    createTrackbar("numLevels", windowName, &numLevels, 100, NULL);
    createTrackbar("prior", windowName, &prior, 10, NULL);
    createTrackbar("histogramBins", windowName, &histogramBins, 15, NULL);
    createTrackbar("numIter", windowName, &numIterations, 30, NULL);
    createTrackbar("regionSize", windowName, &regionSize, 200, NULL);
    createTrackbar("ruler", windowName, &ruler, 255, NULL);
}

void pointCloudTrackbars() {
    String windowName = "Pt. Cloud Trackbars";

    namedWindow(windowName, 0);
    resizeWindow(windowName, 400, 200);
    createTrackbar("X", windowName, &xPtCloud, 1000, NULL);
    createTrackbar("Y", windowName, &yPtCloud, 1000, NULL);
    createTrackbar("Z", windowName, &zPtCloud, 1000, NULL);
    createTrackbar("Scale", windowName, &scale, 100, NULL);
}
////////////////////////////////////////////////////////////

void superpixels() {
    seeds = ximgproc::createSuperpixelSLIC(cimgL, ximgproc::SLIC, ((regionSize>0) ? regionSize : regionSize + 1), ruler);
    //seeds->enforceLabelConnectivity(1);
    
    cout << "Starting iterations" << endl;
    seeds->iterate(numIterations);
    seeds->getLabels(labels);
    seeds->getLabelContourMask(superpixelEdges, false);
    
    //////////////////////////////////////////////////////////////////
    //To show the superpixelated img. Only needed for debug purposes//
    superpixelatedImg = cimgL.clone();
    superpixelatedImg.setTo(Scalar(0, 255, 0), superpixelEdges);
    //////////////////////////////////////////////////////////////////

    maxLabel = seeds->getNumberOfSuperpixels();
    if (!useFakeBackground)
        filteredImg = Mat::zeros(cimgL.size().width, cimgL.size().height, cimgL.type());
    else
        filteredImg = fakeBackgroundImg.clone();

    //Iterates through every superpixel
    Mat validPointsMask = Mat::zeros(cimgL.size(), CV_8UC1);
    Mat ones = Mat::ones(cimgL.size(), CV_8UC1);
    for (int labelNum = 0; labelNum <= maxLabel; labelNum++) {

        //Masks out everything but 1 superpixel at a time
        labelMask = labels == labelNum;
        double avg = mean(thresh, labelMask)[0];

        //Only copies over superpixels whose avg. on the disp8U image is > thresh
        if (avg >= FAR_AVG_THRESH)
            //cimgL.copyTo(filteredImg, labelMask);
            ones.copyTo(validPointsMask, labelMask);
    }
    validPointsMask *= 255;
    cimgL.copyTo(filteredImg, validPointsMask);
    disp8U_filtered.release();
    disp8U.copyTo(disp8U_filtered, validPointsMask);
}

void findDisparity(Mat Limg, Mat Rimg) {
    //Parameters are set by calling init_sbm()
    //sbm->setMinDisparity(-minDisparity);
    sbm->compute(Limg, Rimg, disp16S);

    ///////////////////////////////
    //Mat disp16S_2, disp8U_2, diff;
   /* Mat disp8U_orig, disp8U_2_orig;*/
    //Mat disp8U_crop, disp8U_2_crop;

    //cv::flip(Limg, Limg, 1);
    //cv::flip(Rimg, Rimg, 1);


    //sbm->compute(Rimg, Limg, disp16S_2);

    //cv::flip(disp16S_2, disp16S_2, 1);
    //absdiff(disp16S, disp16S, diff);

    //imshow("DIFF", diff);

    //sbm->setMinDisparity(-minDisparity);
    //sbm->compute(Rimg, Limg, disp16S);
    disp16S.convertTo(disp8U, CV_8UC1, 255 / (ndisparities*16.0));
    //disp16S_2.convertTo(disp8U_2, CV_8UC1, 255 / (ndisparities*16.0));
    //imshow("disp8U_2", disp8U_2);
    //Rect crop1 = Rect(63, 0, disp8U.cols-63-68, disp8U.rows);
    //Rect crop2 = Rect(ndisparities, 0, disp8U_2.cols - 63 - 68, disp8U_2.rows);
    //disp8U_crop = disp8U(crop1);
    //disp8U_2_crop = disp8U_2(crop2);
    //absdiff(disp8U_crop, disp8U_2_crop, diff);
    //imshow("DIFF", diff);
    //waitKey(30);
}

bool readMats(){
    FileStorage fsIntr(INTRINSICS_FILE_PATH, FileStorage::READ);
    FileStorage fsExtr(EXTRINSICS_FILE_PATH, FileStorage::READ);

    //TODO: Do something if it does not find the extrinsics and intrinsic .yml files 
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

    //Convert to Grayscale
    cvtColor(cimgL, crL, COLOR_BGR2GRAY);
    cvtColor(cimgR, crR, COLOR_BGR2GRAY);

    GaussianBlur(crL, crL, Size(3, 3), 0, 0, BORDER_DEFAULT);
    GaussianBlur(crR, crR, Size(3, 3), 0, 0, BORDER_DEFAULT);
    
    equalizeHist(crL, crL);
    equalizeHist(crR, crR);

    //Double contrast to improve disp. performance
    //crL.convertTo(crL, -1, 2, 0);
    //crR.convertTo(crR, -1, 2, 0);
}

void postProc (){
    //imshow("BEFOREBEFORE", disp16S);
    //filterSpeckles(disp16S, 0, 5, 4);
    //imshow("AFTERAFTER", disp16S);
    //Mat temp, kernel;
    /*erode(thresh, temp, kernel);
    dilate(temp, thresh, kernel);*/
    
    //medianBlur(thresh, thresh, 5);        ///Works the best
    //GaussianBlur(thresh, thresh, Size(5, 5), 75, 0, 4);
    //morphologyEx(thresh, thresh, MORPH_GRADIENT, getStructuringElement(MORPH_RECT, Size(7, 7)));
    //floodFill(thresh, Point(FRAME_WIDTH/2, FRAME_HEIGHT/2), Scalar(255));
    //blur(thresh, thresh, Size (5, 5));
    //Mat temp; 
    //bilateralFilter(thresh, temp, 5, 75, 75);
    //imshow("Temp", temp);
    
}

void renderScene() {

    glMatrixMode(GL_PROJECTION);
    glRotatef(currentAngle-angle, 0, 1, 0);
    glMatrixMode(GL_MODELVIEW);
}

void mouseControl(int button, int _x, int _y) {
    if (button == GLUT_LEFT_BUTTON) {
        angle = (_x - xOrigin) * (180.0/ FRAME_WIDTH);  // *180.0/FRAME_WIDTH maps the delta X to a corresponding angle
                                                        //given that left of screen = -90 deg, and right part = 90 deg

        if (angle > 90)
            angle = 90;
        else if (angle < -90)
            angle = -90;
    }
}


void drawPointCloud() {
    printf("Generating point cloud...\n");

    threshold(disp8U_filtered, thresh_3D, FAR_THRESH, 255, THRESH_TOZERO);

    pointMat = Mat(disp16S.size(), CV_32FC3);

    reprojectImageTo3D(thresh_3D, pointMat, Q, true, -1); //pointMat type: CV_32FC3, pointMat channels: 3
}

void display()
{
    xOrigin = FRAME_WIDTH / 2.0;
    if (WM_MOUSEMOVE) {
        POINT p;
        GetCursorPos(&p);
        wglGetCurrentDC();
        ScreenToClient(WindowFromDC(wglGetCurrentDC()), &p);

        button = (GetKeyState(VK_LBUTTON) & 0x100) ? GLUT_LEFT_BUTTON : -1;
       

        int _x = p.x;
        int _y = p.y;
        mouseControl(button, _x, _y);
        cout << "\n~~~~~~~~~~~~~~~~~~~~~\nButton:" << button << "  Angle: " << angle << "  _x: " << _x << "\n~~~~~~~~~~~~~~~~~~~~~\n";//<< "Y:" << state
        renderScene();
        currentAngle = angle;
    }

    /////////////////////////////////////
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glPointSize(1);
    glColor3ub(0, 100, 0);
    glutSolidSphere(0.5, 20, 20);

    glBegin(GL_POINTS); // render with points

    drawPointCloud();

    int negatives = 50;
    validPtCount = 0;
    for (int row = 0; row < pointMat.size().height; row++)
    {
        for (int col = 0; col < pointMat.size().width; col++)
        {
            if (abs(pointMat.at<Point3f>(row, col).x) < 1000 && abs(pointMat.at<Point3f>(row, col).y) < 1000 && abs(pointMat.at<Point3f>(row, col).z) < 1000) {
                Point3f pt(pointMat.at<Point3f>(row, col).x*scaleDown - 0, pointMat.at<Point3f>(row, col).y*scaleDown - 0, pointMat.at<Point3f>(row, col).z*scaleDown - negatives);
                glColor3ub (cimgL.at<Vec3b>(row, col)[2], cimgL.at<Vec3b>(row, col)[1], cimgL.at<Vec3b>(row, col)[0]);
                glVertex3f (pointMat.at<Point3f>(row, col).x*scaleDown - 0, pointMat.at<Point3f>(row, col).y*scaleDown - 0, pointMat.at<Point3f>(row, col).z*scaleDown - negatives);
                validPtCount++;
            }
        }
    }
    
    glPopMatrix();
    glPopAttrib();

    glEnd();
    /////////////////////////////////////
    glFlush();
    glutSwapBuffers();
}

void writeCloudToFile(char* name) {
    cout << "Saving Point Cloud to File.\n";
    std::ofstream fout("ptptCloudTest.ply");

    fout << "ply\n";
    fout << "format ascii 1.0\n";
    fout << "comment made by Oswaldo Ferro\n";
    fout << "comment This contains a Point Cloud\n";
    fout << "element vertex " << validPtCount << "\n";
    fout << "property float x\n";
    fout << "property float y\n";
    fout << "property float z\n";
    fout << "property uchar blue\n";
    fout << "property uchar green\n";
    fout << "property uchar red\n";
    fout << "end_header\n";


    int negatives = 50;
    for (int row = 0; row < pointMat.size().height; row++)
    {
        for (int col = 0; col < pointMat.size().width; col++)
        {
            if (abs(pointMat.at<Point3f>(row, col).x) < 1000 && abs(pointMat.at<Point3f>(row, col).y) < 1000 && abs(pointMat.at<Point3f>(row, col).z) < 1000) {
                Point3f pt(pointMat.at<Point3f>(row, col).x*scaleDown - 0, pointMat.at<Point3f>(row, col).y*scaleDown - 0, pointMat.at<Point3f>(row, col).z*scaleDown - negatives);
                int colour[3] = { cimgL.at<Vec3b>(row, col)[0], cimgL.at<Vec3b>(row, col)[1], cimgL.at<Vec3b>(row, col)[2] };
                fout <<pt.x << " " << pt.y << " " << pt.z
                     << " " << colour[0] << " " << colour[1] << " " << colour[2]
                     << "\n";
            }
        }
    }

    fout.close();
    cout << "Point Cloud Saved to File.\n";
}

void mainLoop() {
    if (webcam) { 
        ////READ
        rawL = PointGreyCam->get_raw_data_force_update();
        rawR = PointGreyCam2->get_raw_data_force_update();
        if (rawL == NULL || rawR == NULL) {
            cout << "Failed to read raw";
            endProgram();
            return;
            waitKey(0);
        }

        ////DEMOSAIC
        imgBayerL = Mat(FRAME_HEIGHT, FRAME_WIDTH, MAT_CONVERSION_CHANNELS, rawL, Mat::AUTO_STEP);
        imgBayerR = Mat(FRAME_HEIGHT, FRAME_WIDTH, MAT_CONVERSION_CHANNELS, rawR, Mat::AUTO_STEP);


        if (imgBayerL.empty() || imgBayerR.empty()) {
            cout << "End of video file reached. Exiting...\n";
            waitKey(0);
            return;
        }

        if (imgBayerL.channels() == 1) {
            cvtColor(imgBayerL, cimgL, COLOR_BayerBG2BGR);
            cvtColor(imgBayerR, cimgR, COLOR_BayerBG2BGR);
        }
        else {
            imgBayerL.copyTo(cimgL);
            imgBayerR.copyTo(cimgR);
        }

        remap(cimgL, cimgL, rmap[0][0], rmap[0][1], INTER_LINEAR);
        remap(cimgR, cimgR, rmap[1][0], rmap[1][1], INTER_LINEAR);

        //Cropping to ROI size
        cimgL = cimgL(newRoi);
        cimgR = cimgR(newRoi);

    }
    else {
        cimgL = imread(imgLfn, IMREAD_COLOR);
        cimgR = imread(imgRfn, IMREAD_COLOR);
    }

    imshow("L", cimgL);
    waitKey(30);

    //FIND DISPARITY

    ////////////Displaying Rectified Images side by side (debugging)/////////
    /*Mat H;
    hconcat(cimgL, cimgR, H);

    int distBtwnLines = 20;
    for (int l = 0; l < H.rows; l += distBtwnLines)
    line(H, Point(0, l), Point(H.cols, l), Scalar(0, 0, 255));
    rectangle(H, roiL, Scalar(0, 0, 255), 2, 8, 0);
    rectangle(H, Rect(roiR.x + cimgL.cols, roiR.y, roiR.width, roiR.height), Scalar(0, 0, 255), 2, 8, 0);
    rectangle(H, newRoi, Scalar(0, 255, 0), 2, 8, 0);
    imshow("Combo", H);*/
    /////////////////////////////////////////////////////////////////////////

    //Readying imgs to find disparity
    if (preProcess)
        preProc();

    //Outputs disparity map to disp8U
    findDisparity(crL, crR);

    //threshold(disp8U, threshTemp, CLOSE_THRESH, 0, 4);    //Close cut-off plane
    threshold(disp8U, thresh, FAR_THRESH, 255, THRESH_BINARY);//THRESH_BINARY);// | THRESH_OTSU);    //The far plane is omitted if using OTSU flag

                                                              //Use background screenshot to improve disparity map
    if (useRealBackground) {
        absdiff(realBackground, cimgL, diff_BG_FG);

        cvtColor(diff_BG_FG, diff_BG_FG, COLOR_BGR2GRAY);
        threshold(diff_BG_FG, diff_BG_FG, 10, 200, THRESH_BINARY);

        erode(diff_BG_FG, diff_BG_FG, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        thresh += diff_BG_FG;
    }

    //Find Superpixels
    superpixels();

    //Attempts at post processing threshold
    if (postProcess)
        postProc();

    //OUTPUT IMGS
    if (showDebugImgs) {
        imshow("disp8U", disp8U);
        imshow("Superpixels!", superpixelatedImg);
        imshow("Thresh", thresh);
        imshow("cimgL", cimgL);
        imshow("disp8U NEW", disp8U_filtered);

        //For comparison purposes only
        //Old version of masking
        //maskedL = Mat::zeros(cimgL.size().width, cimgL.size().height, cimgL.type());
        //maskedR = Mat::zeros(cimgL.size().width, cimgL.size().height, cimgL.type());
        //cimgL.copyTo(maskedL, thresh);
        //cimgR.copyTo(maskedR, thresh);
        //imshow("MaskedL", maskedL);
    }
    imshow("FilteredImg", filteredImg);
    waitKey(30);

    if (show3D) {
        display();
    }

    //SAVE DEBUG IMAGES
    if (saveDebugImgs) {
        VWdisp8U.open(string(PATH_TO_VIDEOS) + "disp8U.avi", 0, POINT_GREY_FPS, disp8U.size(), false);
        VWsuperpixelatedImg.open(string(PATH_TO_VIDEOS) + "Superpixelated.avi", 0, POINT_GREY_FPS, superpixelatedImg.size(), false);
        VWthresh.open(string(PATH_TO_VIDEOS) + "Thresh.avi", 0, POINT_GREY_FPS, thresh.size(), false);

        VWdisp8U << disp8U;
        VWsuperpixelatedImg << superpixelatedImg;
        VWthresh << thresh;
    }

    //USER INPUT - Saving, Pausing and Ending
    key = waitKey(2);

    switch (key) {
    case ESC_KEY:    //END PROGRAM
        endProgram();
        return;
        break;

    case 's':    //STEP THROUGH PAUSED VIDEO
        printf("Stepping into next frame...");
        PointGreyCam->start_capture();
        PointGreyCam2->start_capture();

        rawL = PointGreyCam->get_raw_data_force_update();
        rawR = PointGreyCam2->get_raw_data_force_update();

        PointGreyCam->stop_capture();
        PointGreyCam2->stop_capture();

        break;

    case 'b':    //CAPTURE BACKGROUND
        printf("Capturing background...");

        imshow("CIMGL", cimgL);
        waitKey(30);
        realBackground = cimgL;
        imwrite("DebugImgs/background.bmp", realBackground);
        useRealBackground = true;
        break;

    case 'c':  //SAVES POINT CLOUD TO .OBJ FILE
        writeCloudToFile("pointCloud");


        PointWriter.release();
        printf("Point cloud has been saved!\n");
        break;

    case 't': //TOGGLES USE REAL BACKGROUND
        if (useRealBackground)
            useRealBackground = false;
        else
            useRealBackground = true;
        break;
    case 'p':    //PAUSE PROGRAM
        if (!paused) {
            printf("===============================\n");
            printf("Capture Paused - Press 'p' to continue capturing\n");
            PointGreyCam->stop_capture();
            PointGreyCam2->stop_capture();
            paused = true;
        }
        else {
            printf("===============================\n");
            printf("Resuming Capture...\n");
            PointGreyCam->start_capture();
            PointGreyCam2->start_capture();
            paused = false;
        }
        break;
    }
    //////////////////////////////////////////////////////////User input - end
}

void init_openGL(int argc, char** argv) {

    glutInit(&argc, argv);  //Can only be initialised once
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition(10, 10);
    glutInitWindowSize(FRAME_WIDTH, FRAME_HEIGHT);
    glutCreateWindow("PointCloud");

    //////////////////////////////////////////////////
    //init
    glClearColor(1.0, 1.0, 1.0, 0.0);

    glColor3f(0.0, 1.0, 0.0);
    glPointSize(10);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(48, 1.333, 0.01, 100);
    glRotatef(180, 0, 0, 1);
    glRotatef(180, 0, 1, 0);
    glTranslatef(0, 0, 20);
    glTranslatef(0, 0, 40);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////MAIN/////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    if (argc == 0) {
        cout << "Not enough arguments" << endl;
        return -1;
    }

    CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1.0|}{nr||}{help||}{@input|../data/stereo_calib.xml|}{iL|Images/meL-1meter.png|}{iR|Images/meR-1meter.png|}{fakeBackground|Images/moonBackground1.jpg|}");
    imgLfn = parser.get<string>("iL");
    imgRfn = parser.get<string>("iR");

    //Read in images for background comparison
    String fakeBackgroundImgfn = parser.get<string>("fakeBackground");
    if (fakeBackgroundImgfn == "none")
        useFakeBackground = false;
    if (cvHaveImageReader("DebugImgs/background.bmp")) {
        realBackground = imread("DebugImgs/background.bmp", IMREAD_COLOR);
    }

    //Read in intrinsic and extrinsic matrices from calibration    
    if (!readMats())
        return false;

    //Initialize disparity parameters
    init_sbm(1, nullptr);

    //Initialize OpenGL for the Pt. Cloud 3D rendering
    if (show3D) {
        init_openGL(argc, argv);
    }

    if (!webcam) {
        cimgL = imread(imgLfn, IMREAD_COLOR);
        cimgR = imread(imgRfn, IMREAD_COLOR);

        disp16S = Mat(cimgL.rows, cimgL.cols, CV_16S);
        disp8U = Mat(cimgL.rows, cimgL.cols, CV_8UC1);

        if (cimgL.empty() || cimgR.empty()) {
            cout << "Could not load image";
            return -1;
        }
        imageSize = cimgL.size();
        FRAME_WIDTH = imageSize.width;
        FRAME_HEIGHT = imageSize.height;
    }
    else {
        if (init_PointGrey()) {
            cout << "Unable to start camera(s)";
            waitKey(0);
            return 1;
        }
        
        disp16S = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_16S);
        disp8U = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);

        PointGreyCam->set_trigger_mode(false);
        PointGreyCam2->set_trigger_mode(false);
        
        //Initial read/demosaic
        do {
            rawL = PointGreyCam->get_raw_data_force_update();
            rawR = PointGreyCam2->get_raw_data_force_update();
            cout << "READ!\n";
        } while (!rawL || !rawR);

        imgBayerL = Mat(FRAME_HEIGHT, FRAME_WIDTH, MAT_CONVERSION_CHANNELS, rawL, Mat::AUTO_STEP);
        imgBayerR = Mat(FRAME_HEIGHT, FRAME_WIDTH, MAT_CONVERSION_CHANNELS, rawR, Mat::AUTO_STEP);

        if (imgBayerL.channels() == 1) {
            cvtColor(imgBayerL, cimgL, COLOR_BayerRG2BGR);
            cvtColor(imgBayerR, cimgR, COLOR_BayerRG2BGR);
        }
        else {
            imgBayerL.copyTo(cimgL);
            imgBayerR.copyTo(cimgR);
        }

        //Checks that both cameras are reading in properly
        if (!cimgR.empty() && cimgL.size() == cimgR.size())
            imageSize = cimgL.size();
    }
    
    //DISPLAYS TRACKBARS
    //maskingTrackbars();
    disparityTrackbars();
    superPixelTrackbars();
    threshTrackbars();
    pointCloudTrackbars();

    //CALCULATES RECTIFICATION MAP
    stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &roiL, &roiR);
    initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    newRoi = roiL & roiR;    //Intersection of both ROIs

    /////////////////////Background Img///////////////////////
    if (useFakeBackground) {
        fakeBackgroundImg = imread(fakeBackgroundImgfn, IMREAD_COLOR);
        fakeBackgroundImg = fakeBackgroundImg(newRoi);
    }
    //////////////////////////////////////////////////////////
   
    //disp16S.convertTo(disp32f, CV_32FC3);
    //
    //reprojectImageTo3D(disp32f, pointMat, Q, true, -1); //pointMat type: CV_32FC3, pointMat channels: 3
    //cloudWidget = viz::WCloud(pointMat, viz::Color::green());

    //MAIN LOOP: Read, Find Disp, Superpixellate, Filter, Display 3D

    if (webcam) {   //Start with a paused frame
        printf("===============================\n");
        printf("Capture Paused - Press 'p' to continue capturing\n");
        PointGreyCam->stop_capture();
        PointGreyCam2->stop_capture();
        paused = true;
    }

    while (true) {
        mainLoop();
        //glutMainLoop();
    }

    cout << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\n~~~~~~~~~~~~~~~~~~~~~~~END OF PROGRAM~~~~~~~~~~~~~~~~~~~~~~"
         << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\nExiting...";
    Sleep (3000);
}