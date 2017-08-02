﻿//Oswaldo Ferro - Interaptix

#include <opencv2/stereo/stereo.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <stdio.h>
#include <iostream>
#include<fstream>
#include <iomanip>
#include <opencv2/aruco.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/seeds.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "point_grey_cam.h"
#include "point_grey_sim.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
//#include "opencv2/viz.hpp"
//#include "opencv2/viz/widgets.hpp"

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
Mat disp32f;
//viz::Viz3d ModelWindow("Point cloud with colour");
//viz::WCloud pointCloud = Mat(1,1,;
//viz::WCloud cloudWidget(Mat::zeros(865, 1160, CV_32FC3), viz::Color::green());
std::vector<Point3f> filteredPoints;
std::vector<Vec3b> colour_vector;
bool show3D = true;
bool alreadyRan = false;

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
//int eyeX = 36 / 2 * 10;
//int eyeY = 22 / 2 * 10;
//int eyeZ = 70 / 2 * 10;
//int centerX= 0 * 10;
//int centerY= 0 * 10;
//int centerZ= 0 * 10;
//int upX    = 0 * 10;
//int upY    = 0 * 10;
//int upZ = 1 * 10;

int x = 500, y = 500, z = 500;
int scale = 1;
float scaleDown = 0.1;

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

void mouseCloudControl() {
    // position
    glm::vec3 position = glm::vec3(0, 0, 5);
    // horizontal angle : toward -Z
    float horizontalAngle = 3.14f;
    // vertical angle : 0, look at the horizon
    float verticalAngle = 0.0f;
    // Initial Field of View
    float initialFoV = 45.0f;

    float speed = 3.0f; // 3 units / second
    float mouseSpeed = 0.005f;

    // Get mouse position
    double xpos, ypos;
    //GLFWwindow* window = glfwCreateWindow(FRAME_WIDTH, FRAME_HEIGHT, "Point Cloud New Version", NULL, NULL);

    //glfwGetCursorPos(window, &xpos, &ypos);
    //glfw::glfwGetMousePos(&xpos, &ypos);
    // Reset mouse position for next frame
    //glfwSetCursorPos(window, FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0);
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

void init_sbm() {
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
    createTrackbar("FAR_AVG_THRESH", windowName, &FAR_AVG_THRESH, 255, NULL);
}

void superPixelTrackbars() {
    //int numSuperpixels = 500, numLevels = 4, prior = 2, histogramBins = 5, numIterations = 8, maxLabel;

    String windowName = "superPX Trackbars";

    namedWindow(windowName, 0);
    createTrackbar("numSuperpx", windowName, &numSuperpixels, 1000, NULL);
    createTrackbar("numLevels", windowName, &numLevels, 100, NULL);
    createTrackbar("prior", windowName, &prior, 10, NULL);
    createTrackbar("histogramBins", windowName, &histogramBins, 15, NULL);
    createTrackbar("numIter", windowName, &numIterations, 30, NULL);
    createTrackbar("regionSize", windowName, &regionSize, 200, NULL);
    createTrackbar("ruler", windowName, &ruler, 255, NULL);
}

void pointCloudTrackbars() {
    //int numSuperpixels = 500, numLevels = 4, prior = 2, histogramBins = 5, numIterations = 8, maxLabel;

    String windowName = "Pt. Cloud Trackbars";

    namedWindow(windowName, 0);
    //createTrackbar("eyeX", windowName, &eyeX, 1000, NULL);
    //createTrackbar("eyeY", windowName, &eyeY, 100, NULL);
    //createTrackbar("eyeZ", windowName, &eyeZ, 100, NULL);
    //createTrackbar("centerX", windowName, &centerX, 1000, NULL);
    //createTrackbar("centerY", windowName, &centerY,1000, NULL);
    //createTrackbar("centerZ", windowName, &centerZ, 1000, NULL);
    //createTrackbar("upX", windowName, &upX, 1000, NULL);
    //createTrackbar("upY", windowName, &upY, 1000, NULL);
    //createTrackbar("upZ", windowName, &upZ, 1000, NULL);

    createTrackbar("X", windowName, &x, 1000, NULL);
    createTrackbar("Y", windowName, &y, 1000, NULL);
    createTrackbar("Z", windowName, &z, 1000, NULL);
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
    for (int labelNum = 0; labelNum <= maxLabel; labelNum++) {

        //Masks out everything but 1 superpixel at a time
        labelMask = labels == labelNum;
        double avg = mean(thresh, labelMask)[0];

        //Only copies over superpixels whose avg. on the disp8U image is > thresh
        if (avg >= FAR_AVG_THRESH)
            cimgL.copyTo(filteredImg, labelMask);
    }
}

void findDisparity(Mat Limg, Mat Rimg) {
    //Parameters are set by calling init_sbm()
    //sbm->setMinDisparity(-minDisparity);
    sbm->compute(Limg, Rimg, disp16S);

    ///////////////////////////////
    Mat disp16S_2, disp8U_2, diff;
   /* Mat disp8U_orig, disp8U_2_orig;*/
    Mat disp8U_crop, disp8U_2_crop;

    cv::flip(Limg, Limg, 1);
    cv::flip(Rimg, Rimg, 1);


    sbm->compute(Rimg, Limg, disp16S_2);

    cv::flip(disp16S_2, disp16S_2, 1);
    //absdiff(disp16S, disp16S, diff);

    //imshow("DIFF", diff);

    //sbm->setMinDisparity(-minDisparity);
    //sbm->compute(Rimg, Limg, disp16S);
    disp16S.convertTo(disp8U, CV_8UC1, 255 / (ndisparities*16.0));
    disp16S_2.convertTo(disp8U_2, CV_8UC1, 255 / (ndisparities*16.0));
    //imshow("disp8U_2", disp8U_2);
    Rect crop1 = Rect(63, 0, disp8U.cols-63-68, disp8U.rows);
    Rect crop2 = Rect(ndisparities, 0, disp8U_2.cols - 63 - 68, disp8U_2.rows);
    disp8U_crop = disp8U(crop1);
    disp8U_2_crop = disp8U_2(crop2);
    absdiff(disp8U_crop, disp8U_2_crop, diff);
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

    imshow("Eq", crL);

    //Double contrast to improve disp. performance
    //crL.convertTo(crL, -1, 2, 0);
    //crR.convertTo(crR, -1, 2, 0);
}

void postProc (){
    imshow("BEFOREBEFORE", disp16S);
    filterSpeckles(disp16S, 0, 5, 4);
    imshow("AFTERAFTER", disp16S);
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

void drawPointCloud() {
    printf("Generating point cloud...\n");

    threshold(disp8U, thresh_3D, FAR_THRESH, 255, THRESH_TOZERO);

    pointMat = Mat(disp16S.size(), CV_32FC3);

    reprojectImageTo3D(thresh_3D, pointMat, Q, true, -1); //pointMat type: CV_32FC3, pointMat channels: 3

    filteredPoints.clear();
    colour_vector.clear();

    int negatives = 50;

    //ofstream of;
    //of.open("PointCloudFile_SCALED_DOWN.txt");
    cout << pointMat.size() << "   " << cimgL.size() << "\n";

    for (int row = 0; row < pointMat.size().height; row++)
        {
            for (int col = 0; col < pointMat.size().width; col++)
            {
                if (abs(pointMat.at<Point3f>(row, col).x) < 1000 && abs(pointMat.at<Point3f>(row, col).y) < 1000 && abs(pointMat.at<Point3f>(row, col).z) < 1000 ) {
                    Point3f pt(pointMat.at<Point3f>(row, col).x*scaleDown - 0, pointMat.at<Point3f>(row, col).y*scaleDown - 0, pointMat.at<Point3f>(row, col).z*scaleDown - negatives);
                    filteredPoints.push_back(pt);
                    //filteredPoints.push_back(pointMat.at<Point3f>(row, col));
                    colour_vector.push_back(cimgL.at<Vec3b>(row, col));
                    //of << "[" << pointMat.at<Point3f>(row, col).x << ", " << pointMat.at<Point3f>(row, col).y << ", " << pointMat.at<Point3f>(row, col).z << "]\n";
                    //of << "[" << pt.x << ", " << pt.y << ", " << pt.z << "]\n";
                }
            }
        }
    //negatives = 50;
    //filteredPoints.push_back(Point3f((x/10.0+0.5-negatives)*scale, (y/10.0+0-negatives)*scale, (z/10.0+0-negatives)*scale+2.0));
    //filteredPoints.push_back(Point3f((x/10.0+0-negatives)*scale, (y/10.0+0-negatives)*scale, (z/10.0+0-negatives)*scale));
    //filteredPoints.push_back(Point3f(0.00666812, -1.81203, -3.97678));
    //filteredPoints.push_back(Point3f(1.0, 1.081203, -6.97678));
}

void display()
{
    //mainLoop();

    /////////////////////////////////////
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    //glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    //gluOrtho2D(0.0, 400.0, 0.0, 150.0);
    //gluPerspective(fov, aspect_3D, nearClipping, farClipping);
    //glViewport(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    //glTranslatef(-35 / 2.0, 0,0);
    //glMultMatrixf(filteredPoints);

    //gluLookAt(3, 4, 2, 0, 0, 0, 0, 0, 1);
    //Mat filteredPointsMat = Mat(filteredPoints);

    /*float minX = -9999999, maxX = -9999999, minY = -9999999, maxY = -9999999, minZ = -9999999, maxZ = -9999999;
    float centreX, centreY, centreZ;
    for (auto i : filteredPoints) {
    if (i.x > maxX) maxX = i.x;
    if (i.x < minX) minX = i.x;
    if (i.y > maxY) maxY = i.y;
    if (i.y < minY) minY = i.y;
    if (i.z > maxZ) maxZ = i.z;
    if (i.z < minZ) minZ = i.z;
    }
    cout    << "\n maxX" << maxX
    <<"\n minX" << minX
    <<"\n maxY" << maxY
    <<"\n minY" << minY
    <<"\n maxZ" << maxZ
    <<"\n minZ" << minZ;*/


    //maxX  35.4924
    //minX   - 1e+07
    //maxY  21.7356
    //minY   - 1e+07
    //maxZ  70.1333
    //minZ   - 1e+07



    //minMaxLoc(filteredPointsMat , minX, maxX);
    //gluLookAt(maxX/2, maxY/2, maxZ/2, 0, 0, 0, 0, 0, 1);
    //gluLookAt(eyeX/10.0, eyeY/10.0, eyeZ/10.0, centerX/10.0, centerY / 10.0, centerZ / 10.0, upX/10.0, upY/10.0, upZ/10.0);

    glPointSize(1);
    //glutSolidSphere(0.5, 20, 20);

    glBegin(GL_POINTS); // render with points

    cout << filteredPoints.size() << "   " << colour_vector.size()<<"\n";
    for (auto i : filteredPoints) {
        //int b = colour_vector.back()[0];
        //cout << "b"<<;
        glColor3b (colour_vector.back()[2], colour_vector.back()[1], colour_vector.back()[0]);
        colour_vector.pop_back();
        glVertex3f(i.x, i.y, i.z);
    }

    glPopMatrix();
    glPopAttrib();

    glEnd();
    /////////////////////////////////////
    glFlush();
    glutSwapBuffers();
    //glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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
        imshow("L", cimgL);
        waitKey(30);
    }


    //FIND DISPARITY

    imshow("L", cimgL);
    waitKey(30);
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
        //cvtColor(realBackground, realBackground, CV_BGR2HSV);
        //cvtColor(cimgL, cimgL, CV_BGR2HSV);

        absdiff(realBackground, cimgL, diff_BG_FG);
        //cvtColor(cimgL, cimgL, CV_HSV2BGR);
        cvtColor(diff_BG_FG, diff_BG_FG, COLOR_BGR2GRAY);
        threshold(diff_BG_FG, diff_BG_FG, 10, 200, THRESH_BINARY);
        //imshow("thresh-before", thresh);
        erode(diff_BG_FG, diff_BG_FG, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        thresh += diff_BG_FG;
        //thresh = diff_BG_FG.clone();
        //imshow("thresh-after", thresh);
        //imshow("DIFFERENCE", diff_BG_FG);
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
        drawPointCloud();
        display();
    }
    //ModelWindow.spinOnce(30);

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

    case 'c':  //GENERATES POINT CLOUD
        if (show3D) {
            show3D = false;
            printf("show3D is now false");
        }
        else {
            show3D = true;
            printf("show3D is now true");
        }

        if (PointWriter.isOpened() && !pointMat.empty()) {
            //cout << "Channels: " << pointMat.channels();
            //PointWriter << "PointMat" << pointMat;
            //imwrite("pointMat.jpg", pointMat);

            /*for (int col = 0; col < pointMat.cols; col++) {
            for (int row = 0; row < pointMat.rows; row++) {
            pointMat
            }
            }*/

            cout << "DID NOT SAVE THIS IS EMPTY!!!\n";
        }
        else {
            printf("Unable to open PointWriter file || pointMat is empty\n");
            break;
        }
        //cout << pointMat;
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
//
//void reshape(int x, int y) {
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    //glFrustum(-1.0, 1.0, -1.0, 1.0, 0.0001f, 10000.0);
//    gluPerspective(45, x / y, 0.001f, 100.0f);
//}
//
//void glutIdleFunc(void(*func) (void)) {
//    cout << "Idle...";
//}

void init_openGL(int argc, char** argv) {

    glutInit(&argc, argv);  //Can only be initialised once
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition(10, 10);
    glutInitWindowSize(FRAME_WIDTH, FRAME_HEIGHT);
    //GLFWwindow* window = glfwCreateWindow(FRAME_WIDTH, FRAME_HEIGHT, "PointCloud", NULL, NULL);
    glutCreateWindow("PointCloud");
    //GLFWwindow* window = glfwCreateWindow(FRAME_WIDTH, FRAME_HEIGHT, "Point Cloud New Version", NULL, NULL);

    //////////////////////////////////////////////////
    //init
    glClearColor(1.0, 1.0, 1.0, 0.0);

    glColor3f(0.0, 1.0, 0.0);
    glPointSize(10);
    //glShadeModel(GL_FLAT);
    //glViewPort(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glFrustum(-1.0, 1.0, -1.0, 1.0, 0.0001f, 10000.0);
    //gluPerspective(45, FRAME_WIDTH / FRAME_HEIGHT, 0.001f, 100.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, 1.333, 0.01, 100);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT);
    //gluOrtho2D(0.0, 400.0, 0.0, 150.0);
    //////////////////////////////////////////////////
    //reshape
    //glutDisplayFunc(display);
    //glutReshapeFunc(reshape);
    //glutIdleFunc(glutIdleFunc);
    //glutPostRedisplay();
    //glutMainLoop();
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
    String fakeBackgroundImgfn = parser.get<string>("fakeBackground");
    if (fakeBackgroundImgfn == "none")
        useFakeBackground = false;
    if (cvHaveImageReader("DebugImgs/background.bmp")) {
        //useRealBackground = true;
        realBackground = imread("DebugImgs/background.bmp", IMREAD_COLOR);
    }
    //Read in intrinsic and extrinsic matrices from calibration    
    if (!readMats())
        return false;
    //Initialize disparity parameters
    init_sbm();

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

    //MAIN LOOP: Read, Demosaic, find Disp, Mask
    while (true) {
        mainLoop();
    }

    cout << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\n~~~~~~~~~~~~~~~~~~~~~~~END OF PROGRAM~~~~~~~~~~~~~~~~~~~~~~"
         << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\n=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
         << "\nExiting...";
    Sleep (3000);
}