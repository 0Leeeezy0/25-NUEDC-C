#include "main.h"
#include "process.h"

using namespace cv;
using namespace std;
using namespace ge;

Camera USBCamera;
// 图像
Mat ImgColor;
Mat ImgGray;
Mat ImgBinary;
Mat ImgEdge;
Mat ImgROI;
Mat ImgNano;
Mat ImgTest;
// 图像处理
Image JetsonImage;
Image::BGR JetsonImageBGR;
// 轮廓及其相关
vector<vector<Point>> allContours;
vector<RotatedRect> allRect;
vector<Point> allCenter;
vector<Vec4i> allHierarchy;
vector<vector<Point>> filterContours;
vector<RotatedRect> filterRect;
vector<Point> filterCenter;
vector<vector<Point>> filterContoursNMS;
vector<RotatedRect> filterRectNMS;
vector<Point> filterCenterNMS;
vector<Point2f> quadsPoints;
vector<Point2f> quadsOrderPoints;
vector<double> quadsEdgeLength;         // 边框像素边长 点1-2， 点2-3， 点3-4， 点4-1
double quadsEdgePerimeter;              // 边框像素周长
vector<vector<Point>> innerContours;    // 内部图形轮廓
vector<RotatedRect> innerRect;          // 内部图形轮廓最小外接矩形
vector<Vec4i> innerHierarchy;           // 内部图形索引
vector<vector<Point2f>> innerPoints;    // 内部图形角点
vector<vector<bool>> innerPointsKind;   // 内部图形角点类型（正常边框角点：true；两矩形相交角点：false）
vector<vector<Point2f>> innerSplitPoints;           // 内部分割图形角点
vector<vector<Point2f>> innerSplitTruePoints;       // 内部分割图形矩形边框角点
vector<vector<bool>> innerSplitPointsKind;          // 内部分割图形角点类型（正常边框角点：true；两矩形相交角点：false）
double paperDistanceRaw;
double paperDistanceCalibration;
double paperYaw;
double paperPitch;
double paperRoll;
bool isOverlap;             // 重叠标志位
Mat rvec;   // 旋转向量
Mat tvec;   // 平移向量
// double paperDistanceOffset[11] = {269.0, 293.0, 297, 330, 344, 373, 395, 435, 439, 462, 483}; // 1000mm-2000mm处偏移，步进值100mm
vector<int> innerShape;                     // 内部图形形状（0：三角形；1：正方形；2：圆形）
vector<double> innerShapeEdgeLength;        // 内部图形边长/直径（像素）
int neededNum;                              // 指定的数字
int aiDetectionNum;                         // AI识别的数字
vector<Rect> aiDetectionROI;                // AI识别的ROI
vector<pair<int, int>> aiDetectionNum2Idx;  // AI结果和框索引的映射

// 相机内参矩阵
Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<   585.574746577337, 0, 326.5194804292524,
                                                0, 598.9625112983059, 226.9552532525535,
                                                0, 0, 1);
// 相机畸变系数
Mat distCoeffs = (cv::Mat_<double>(5, 1) <<     0.684432958966514, 41.33898529048504, 0.001877599764388113, -0.001342504805413938, 0);

// 偏移量
double paperDistanceOffset[1+1000/OFFSET_STEP_DISTANCE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // 1000mm-2000mm处偏移，步进值100mm

int main()
{
    unsigned long int frameNum = 0;
    USBCamera.initInfosGet("/dev/video0",V4L2,FPS_60,EXP_6,false,MJPG,WIDTH_640,HEIGHT_480,false);
    USBCamera.Init();

    while(1)
    {
        cout << "\033[2J\033[1;1H"; // 清屏并移动光标到左上角
        
        struct timespec startTime;  // 构造时间戳
        clock_gettime(CLOCK_MONOTONIC, &startTime); // 获取开始时间戳
        // 图像预处理
        imgProcess();

        // 初始化所有变量
        initVariable(); 
debugPoint(1);
        // 查找轮廓
        JetsonImage.findContours(ImgBinary, allContours, allHierarchy, RETR_TREE, CHAIN_APPROX_NONE);   // 存储所有点
debugPoint(2);
        // 筛选轮廓
        contourFilter(allContours);
debugPoint(3);
        // 轮廓非极大值抑制
        contourNMS(filterContours);
debugPoint(4);
        // 绘制所有轮廓
        // Point2f rectLine[4];
        // for(long unsigned int i = 0;i < allRect.size();i++)
        // {
        //     allRect[i].points(rectLine);
        //     for(int k=0;k<4;k++)
        //     {
        //     line(ImgNano,rectLine[k],rectLine[(k+1)%4],Scalar(255,0,0),2);
        //     }
        // }
        // 绘制筛选轮廓
        // for(long unsigned int i = 0;i < filterRectNMS.size();i++)
        // {
        //     filterRectNMS[i].points(rectLine);
        //     for(int k=0;k<4;k++)
        //     {
        //     line(ImgNano,rectLine[k],rectLine[(k+1)%4],Scalar(0,255,0),2);
        //     }
        // }
        // 绘制框轮廓
        JetsonImage.drawContours(ImgNano, filterContoursNMS, allHierarchy, Scalar(0,0,255));
debugPoint(5);
        // 将轮廓转为四边形四个点
        Contours2Quads(filterContoursNMS, filterCenterNMS);
debugPoint(6);
        // 绘制轮廓、角点、序号、边长
        for(long unsigned int i = 0;i < quadsOrderPoints.size();i++)
        {
            circle(ImgNano, quadsOrderPoints[i], 10, cv::Scalar(0, 0, 255), 1);
            string distanceString = to_string(quadsEdgeLength[i]);
            putText(ImgNano,distanceString,Point((quadsOrderPoints[(i+1)%4].x+quadsOrderPoints[i].x)/2,(quadsOrderPoints[(i+1)%4].y+quadsOrderPoints[i].y)/2),FONT_ITALIC,0.5,Scalar(0,255,0));
            switch(i){
                case 0:{ putText(ImgNano,"1",quadsOrderPoints[i],FONT_ITALIC,0.5,Scalar(0,255,0)); break; }
                case 1:{ putText(ImgNano,"2",quadsOrderPoints[i],FONT_ITALIC,0.5,Scalar(0,255,0)); break; }
                case 2:{ putText(ImgNano,"3",quadsOrderPoints[i],FONT_ITALIC,0.5,Scalar(0,255,0)); break; }
                case 3:{ putText(ImgNano,"4",quadsOrderPoints[i],FONT_ITALIC,0.5,Scalar(0,255,0)); break; }
            }
            line(ImgNano,quadsOrderPoints[i],quadsOrderPoints[(i+1)%4],Scalar(0,255,0),1);
        }
debugPoint(7);
        // 获取内部图形形状
        innerShapeGet(filterContoursNMS, isOverlap);
debugPoint(8);
        // 绘制内部图形点
        for(long unsigned int i = 0;i < innerPoints.size();i++){
            for(long unsigned int j = 0;j < innerPoints[i].size();j++){
                line(ImgNano,innerPoints[i][j],innerPoints[i][(j+1)%innerPoints[i].size()],Scalar(0,255,0),1);
                if(isOverlap == true){
                    if(innerPointsKind[i][j] == true)
                        circle(ImgNano, innerPoints[i][j], 6, cv::Scalar(0, 255, 0), 1);
                    else
                        circle(ImgNano, innerPoints[i][j], 6, cv::Scalar(0, 0, 255), 1);
                }
            }
        }
debugPoint(9);
        // // 计算距离与欧拉角
        calu4DistanceAndEularAngle(cameraMatrix, distCoeffs);
debugPoint(10);
        // // 绘制三维坐标轴
        drawCoordinateAxes(ImgNano, cameraMatrix, distCoeffs, rvec, tvec, 200.0f);
debugPoint(11);

        JetsonImage.Add(ImgTest);
        // JetsonImage.Add(ImgBinary);
        // JetsonImage.Add(ImgEdge);
        JetsonImage.Add(ImgROI);
        JetsonImage.Add(ImgNano);
        JetsonImage.Show();

        // JetsonImage.Save(ImgColor,"/home/jetson/Desktop/25-NUEDC-C/GE",1,JPG);

        waitKey(1);

        frameNum++;

        struct timespec endTime;  // 构造时间戳
        clock_gettime(CLOCK_MONOTONIC, &endTime); // 获取开始时间戳

        double FPS = 1.0/((double)(endTime.tv_nsec-startTime.tv_nsec)/((double)1e9)+ (double)(endTime.tv_sec-startTime.tv_sec));  // 计算 FPS
#if PRINT_EN == 1
        cout << "**************************************************************" << endl;
        cout << "FRAME MUN: " << frameNum << "    FPS: " << FPS << endl;            // 帧数 帧率
        cout << "DISTANCE RAW: " << paperDistanceRaw << endl;                       // 原始距离
        cout << "DISTANCE CALIBRATION: " << paperDistanceCalibration << endl;       // 校准距离
        if(isOverlap == true)
            cout << "OVERLAP STATUS: TRUE" << endl;                                 // 重叠情况
        else 
            cout << "OVERLAP STATUS: FALSE" << endl;
        if(quadsEdgeLength.size() == 4 && neededNum == 10){
            for(long unsigned int i = 0;i < innerShape.size();i++)
            {
                // 形状
                switch(innerShape[i]){
                    case 0:{ cout << "SHAPE: TRIANGLE" << endl; break; }            // 三角形
                    case 1:{ cout << "SHAPE: SQUARE" << endl; break; }              // 正方形
                    case 2:{ cout << "SHAPE: CIRCLE" << endl; break; }              // 圆形
                }
                // 边长
                double edgeLength = 2.0f*(CALIBRATION_HEIGHT+CALIBRATION_WIDTH)*innerShapeEdgeLength[i]/quadsEdgePerimeter;
                cout << "EDGE LENGTH: " << edgeLength << endl;      // 边长
                // 面积
                switch(innerShape[i]){
                    case 0:{ cout << "SQUARE: " << edgeLength*edgeLength*sqrt(3)/4 << endl; break; }    // 三角形
                    case 1:{ cout << "SQUARE: " << edgeLength*edgeLength << endl; break; }              // 正方形
                    case 2:{ cout << "SQUARE: " << CV_PI*edgeLength*edgeLength << endl; break; }        // 圆形
                }
            }
        }
        cout << "YAW: " << paperYaw << endl;            // 欧拉角
        cout << "PITCH: " << paperPitch << endl;        // 欧拉角
        cout << "ROLL: " << paperRoll << endl;          // 欧拉角
        cout << "**************************************************************" << endl;
#endif
    }
    return 0;
}