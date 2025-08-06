#ifndef _MAIN_H_
#define _MAIN_H_

#include <iostream>
#include <algorithm> 
#include <vector>
#include <string.h>
#include <math.h>
#include <time.h>
#include "opencv2/opencv.hpp"
#include "ge/camera.h"
#include "ge/control.h"
#include "ge/image.h"
#include "ge/uart.h"
#include "json/json.hpp"

#define IMG_WIDTH 640                       // 图像宽度
#define IMG_HEIGHT 480                      // 图像高度
#define SQUARE_MIN 0                        // 面积筛选最小值
#define SQUARE_MAX 100000                   // 面积筛选最大值
#define CALIBRATION_WIDTH 170.5f            // 标定宽度（mm）
#define CALIBRATION_HEIGHT 260.5f           // 标定高度（mm）
#define OFFSET_STEP_DISTANCE 100            // 偏移标定步长（mm）
#define INNER_SHAPE_SQUARE_MIN 200          // 内部图形面积最小值
#define CHOOSE_CONTOUR_IDX 1                // 选择非极大值抑制轮廓的索引（若外框在最外侧：1，若外框在内测：0）
#define CIRCLE_ROI_RADIUS 50.0f             // 圆形ROI区域半径（坐标筛选ROI）
#define POINT_SORT_CIRCLE_ROI_RADIUS 5      // 圆形ROI区域半径（点分类ROI）

#define DEBUG_EN 0              // DEBUG模式使能 
#define PRINT_EN 1              // 打印模式使能 

extern ge::Camera USBCamera;
extern cv::Mat ImgColor;
extern cv::Mat ImgGray;
extern cv::Mat ImgBinary;
extern cv::Mat ImgEdge;
extern cv::Mat ImgROI;
extern cv::Mat ImgNano;
extern cv::Mat ImgTest;
extern ge::Image JetsonImage;
extern ge::Image::BGR JetsonImageBGR;
extern std::vector<std::vector<cv::Point>> allContours;
extern std::vector<cv::RotatedRect> allRect;
extern std::vector<cv::Point> allCenter;
extern std::vector<cv::Vec4i> allHierarchy;
extern std::vector<std::vector<cv::Point>> filterContours;
extern std::vector<cv::RotatedRect> filterRect;
extern std::vector<cv::Point> filterCenter;
extern std::vector<std::vector<cv::Point>> filterContoursNMS;
extern std::vector<cv::RotatedRect> filterRectNMS;
extern std::vector<cv::Point> filterCenterNMS;
extern std::vector<cv::Point2f> quadsPoints;
extern std::vector<cv::Point2f> quadsOrderPoints;
extern std::vector<double> quadsEdgeLength;     // 点1-2， 点2-3， 点3-4， 点4-1
extern double quadsEdgePerimeter;               // 边框像素周长
extern std::vector<std::vector<cv::Point>> innerContours;   // 内部图形轮廓
extern std::vector<cv::RotatedRect> innerRect;              // 内部图形轮廓最小外接矩形
extern std::vector<cv::Vec4i> innerHierarchy;               // 内部图形索引
extern std::vector<std::vector<cv::Point2f>> innerPoints;   // 内部图形角点
extern std::vector<std::vector<bool>> innerPointsKind;      // 内部图形角点类型（正常边框角点：true；两矩形相交角点：false）
extern std::vector<std::vector<cv::Point2f>> innerSplitPoints;          // 内部分割图形角点
extern std::vector<std::vector<cv::Point2f>> innerSplitTruePoints;      // 内部分割图形矩形边框角点
extern std::vector<std::vector<bool>> innerSplitPointsKind;             // 内部分割图形角点类型（正常边框角点：true；两矩形相交角点：false）
extern double paperDistanceRaw;
extern double paperDistanceCalibration;
extern double paperYaw;
extern double paperPitch;
extern double paperRoll;
extern bool isOverlap;      // 重叠标志位
extern cv::Mat rvec;        // 旋转向量
extern cv::Mat tvec;        // 平移向量
extern cv::Mat cameraMatrix;           // 相机内参矩阵
extern cv::Mat distCoeffs;             // 相机畸变系数
extern double paperDistanceOffset[1+1000/OFFSET_STEP_DISTANCE]; // 1000mm-2000mm处偏移，步进值100mm
extern std::vector<int> innerShape;                             // 内部图形（0：三角形；1：正方形；2：圆形）
extern std::vector<double> innerShapeEdgeLength;                // 内部图形边长/直径（像素）
extern int neededNum;                                           // 指定的数字
extern int aiDetectionNum;                                      // AI识别的数字
extern std::vector<cv::Rect> aiDetectionROI;                    // AI识别的ROI
extern std::vector<std::pair<int, int>> aiDetectionNum2Idx;     // AI结果和框索引的映射

#endif