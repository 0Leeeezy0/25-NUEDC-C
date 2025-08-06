#ifndef _PROCESS_H_
#define _PROCESS_H_

#include <iostream>
#include "opencv2/opencv.hpp"

// 变量初始化
void initVariable(void);

// 图像处理
void imgProcess(void);

// 轮廓筛选
void contourFilter(std::vector<std::vector<cv::Point>> Contours);

// 轮廓非极大值抑制
void contourNMS(std::vector<std::vector<cv::Point>> Contours);

// 轮廓四边形拟合
void Contours2Quads(std::vector<std::vector<cv::Point>> Contours, std::vector<cv::Point> Center);

// 距离和欧拉角计算
void calu4DistanceAndEularAngle(cv::Mat cameraMatrix, cv::Mat distCoeffs);

// 绘制三维坐标轴
void drawCoordinateAxes(cv::Mat& image, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const cv::Mat& rvec, const cv::Mat& tvec, float length = 50.0f);

// 内部形状获取
void innerShapeGet(std::vector<std::vector<cv::Point>> Contours, bool _isOverlap_);

// 重叠矩形提取
void overlapRectExtract(std::vector<cv::Point2f> innerPoint, std::vector<bool> innerPointKind);

// 正方形绘制(对角线点)
void from2PointDrawRotatedSquare(cv::Mat& image, cv::Point2f pt1, cv::Point2f pt2, cv::Scalar color);

// AI识别
void AI(void);

// DEBUG软件打点
void debugPoint(int step);

#endif