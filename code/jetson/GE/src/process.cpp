#include "process.h"
#include "main.h"

using namespace cv;
using namespace std;
using namespace ge;

// 变量初始化
void initVariable(void){
    allContours.clear();
    allRect.clear();
    allCenter.clear();
    allHierarchy.clear();
    filterContours.clear();
    filterRect.clear();
    filterCenter.clear();
    filterContoursNMS.clear();
    filterRectNMS.clear();
    filterCenterNMS.clear();
    quadsPoints.clear();
    quadsOrderPoints.clear();
    quadsEdgeLength.clear();         // 边框像素边长 点1-2， 点2-3， 点3-4， 点4-1
    quadsEdgePerimeter = 0;          // 边框像素周长
    innerContours.clear();    // 内部图形轮廓
    innerHierarchy.clear();   // 内部图形索引
    innerPoints.clear();    // 内部图形轮廓点
    paperDistanceRaw = 0;
    paperDistanceCalibration = 0;
    paperYaw = 0;
    paperPitch = 0;
    paperRoll = 0;
    isOverlap = false;
    rvec = Mat::zeros(ImgColor.size(), CV_64F);   // 旋转向量
    tvec = Mat::zeros(ImgColor.size(), CV_64F);   // 平移向量
    innerShape.clear();             // 内部图形形状（0：三角形；1：正方形；2：圆形）
    innerShapeEdgeLength.clear();   // 内部图形边长/直径（像素）
    neededNum = 10;                 // 指定的数字
    aiDetectionNum = 10;            // 由于不会出现10，所以作为默认值
    aiDetectionROI.clear();         // AI识别ROI
    aiDetectionNum2Idx.clear();     // AI结果和框索引的映射
}

// 图像处理
void imgProcess(void){
    USBCamera.Camera >> ImgColor;

    // 新建画布
    JetsonImage.newImg(ImgROI,WIDTH_640,HEIGHT_480,CV_8UC1,Scalar(0,0,0));
    ImgNano = ImgColor.clone();
    ImgGray = ImgColor.clone();
    ImgTest = ImgColor.clone();
    JetsonImage.colorConvert(ImgGray,COLOR_BGR2GRAY);
    // JetsonImage.Sobel(ImgGray,3);
    // JetsonImage.GaussBlur(ImgGray,5);
    // 图像腐蚀
    // JetsonImage.Erode(ImgGray,4);    // 白色变细
    // JetsonImage.Dilate(ImgGray,3);   // 黑色变细
    JetsonImage.Dilate(ImgGray,3);      // 黑色变细
    ImgBinary = ImgGray.clone();
    // 二值化
    JetsonImageBGR.thresholdConvert(ImgBinary);

    Canny(ImgGray, ImgEdge, 100, 150, 3);

    // 绘制参考线
    line(ImgNano,Point(0,IMG_HEIGHT/2),Point(IMG_WIDTH,IMG_HEIGHT/2),Scalar(255,255,0),1);
    line(ImgNano,Point(IMG_WIDTH/2,0),Point(IMG_WIDTH/2,IMG_HEIGHT),Scalar(255,255,0),1);
    circle(ImgNano, Point(IMG_WIDTH/2,IMG_HEIGHT/2), CIRCLE_ROI_RADIUS, cv::Scalar(255, 255, 0), 1);
}

// 轮廓筛选
void contourFilter(vector<vector<Point>> Contours){
    vector<vector<Point>> squareFilterContours;
    vector<RotatedRect> squareFilterRect;
    vector<Point> squareFilterCenter;

    // 初始化
    filterContours.clear();
    filterRect.clear();
    filterCenter.clear();

    if(allContours.size() != 0){
        JetsonImage.findContoursRect(allContours,allRect);
        JetsonImage.findContoursCenter(allRect,allCenter);
    }

    // 面积筛选
    for(long unsigned int i = 0;i < Contours.size();i++)
    {
        double contourSquare = contourArea(Contours[i]);    // 计算面积
        if (SQUARE_MIN < contourSquare && contourSquare < SQUARE_MAX)
            squareFilterContours.push_back(Contours[i]);
    }
    if(squareFilterContours.size() != 0){
        JetsonImage.findContoursRect(squareFilterContours,squareFilterRect);
        JetsonImage.findContoursCenter(squareFilterRect,squareFilterCenter);
    }
    // 坐标筛选
    for(long unsigned int i = 0;i < squareFilterRect.size();i++)     
    {
        if(sqrt((squareFilterCenter[i].x-IMG_WIDTH/2)*(squareFilterCenter[i].x-IMG_WIDTH/2)+(squareFilterCenter[i].y-IMG_HEIGHT/2)*(squareFilterCenter[i].y-IMG_HEIGHT/2)) < CIRCLE_ROI_RADIUS){  //  && (abs(squareFilterRect[i].angle)-90) < 10
            filterContours.push_back(squareFilterContours[i]);
            filterRect.push_back(squareFilterRect[i]);
            filterCenter.push_back(squareFilterCenter[i]);
        }
    }
    cout << filterCenter.size() << endl;
}

// 非极大值抑制
void contourNMS(vector<vector<Point>> Contours){
    // double maxContourSquare = 0.0;

    // 初始化
    filterContoursNMS.clear();
    filterRectNMS.clear();
    filterCenterNMS.clear();

    if(Contours.size() != 0){
        // 生成面积序列
        vector<pair<double, size_t>> areas;
        for (size_t i = 0; i < Contours.size(); i++) {
            areas.emplace_back(contourArea(Contours[i]), i);
        }

        // 按面积升序排序
        // sort(areas.begin(), areas.end());

        // 按面积降序排序
        sort(areas.begin(), areas.end(), greater<pair<double, size_t>>());

        // 获取排序后的轮廓
        vector<vector<Point>> sortedContours;
        for (const auto& p : areas) {
            sortedContours.push_back(Contours[p.second]);
        }
        // 选择非极大值抑制的值（若有最外框则选择1，若无最外框则选择0）
        if(sortedContours.size() > CHOOSE_CONTOUR_IDX)
            filterContoursNMS.push_back(sortedContours[CHOOSE_CONTOUR_IDX]);
        else if(sortedContours.size() == 1)
            filterContoursNMS.push_back(sortedContours[0]);
        if(filterContoursNMS.size() != 0){
            JetsonImage.findContoursRect(filterContoursNMS,filterRectNMS);
            JetsonImage.findContoursCenter(filterRectNMS,filterCenterNMS);
        }
    }
}

// 轮廓四边形拟合
void Contours2Quads(vector<vector<Point>> Contours, vector<Point> Center){
    if(Contours.size() != 0){
        // 初始化
        quadsOrderPoints.clear();
        quadsEdgeLength.clear();
        // 计算轮廓周长
        quadsEdgePerimeter = cv::arcLength(Contours[0], true);
        // 多边形逼近（epsilon为周长的一定比例）
        approxPolyDP(Contours[0], quadsPoints, 0.02 * quadsEdgePerimeter, true);

        // 角点顺序整定，左上为第一个，顺时针排序
        if(quadsPoints.size() == 4){
            for(int i = 0;i < 4;i++){
                if(quadsPoints[i].x < Center[0].x && quadsPoints[i].y < Center[0].y)
                    quadsOrderPoints.push_back(quadsPoints[i]);    
            }
            for(int i = 0;i < 4;i++){
                if(quadsPoints[i].x > Center[0].x && quadsPoints[i].y < Center[0].y)
                    quadsOrderPoints.push_back(quadsPoints[i]);    
            }
            for(int i = 0;i < 4;i++){
                if(quadsPoints[i].x > Center[0].x && quadsPoints[i].y > Center[0].y)
                    quadsOrderPoints.push_back(quadsPoints[i]);    
            }
            for(int i = 0;i < 4;i++){
                if(quadsPoints[i].x < Center[0].x && quadsPoints[i].y > Center[0].y)
                    quadsOrderPoints.push_back(quadsPoints[i]);    
            }
            if(quadsOrderPoints.size() == 4){
                for(long unsigned int i = 0;i < quadsOrderPoints.size();i++)
                {
                    // 边长计算
                    double distanceDouble = sqrt((quadsOrderPoints[(i+1)%quadsOrderPoints.size()].x-quadsOrderPoints[i].x)*(quadsOrderPoints[(i+1)%quadsOrderPoints.size()].x-quadsOrderPoints[i].x)+(quadsOrderPoints[(i+1)%quadsOrderPoints.size()].y-quadsOrderPoints[i].y)*(quadsOrderPoints[(i+1)%quadsOrderPoints.size()].y-quadsOrderPoints[i].y));
                    quadsEdgeLength.push_back(distanceDouble);
                }
            }
        }
    }
}

// 距离和欧拉角计算
void calu4DistanceAndEularAngle(Mat cameraMatrix, Mat distCoeffs){
    if(quadsOrderPoints.size() == 4){
        // 定义实际3D坐标
        vector<Point3f> objectPoints;
        // 左上角开始顺时针
        objectPoints.push_back(Point3f(-CALIBRATION_WIDTH/2, -CALIBRATION_HEIGHT/2, 0));
        objectPoints.push_back(Point3f(CALIBRATION_WIDTH/2, -CALIBRATION_HEIGHT/2, 0));
        objectPoints.push_back(Point3f(CALIBRATION_WIDTH/2, CALIBRATION_HEIGHT/2, 0));
        objectPoints.push_back(Point3f(-CALIBRATION_WIDTH/2, CALIBRATION_HEIGHT/2, 0));
        // 定义图像坐标
        vector<Point2f> imagePoints;
        // 解PNP
        solvePnP(objectPoints, quadsOrderPoints, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_IPPE);     // 这个。搞死我了，7.30整整一下午啊 

        if(!rvec.empty() && !tvec.empty()){
            // 计算距离（平移向量的Z分量）
            paperDistanceRaw = tvec.at<double>(2);
            // 距离归一化
            double paperDistanceNormalization = (paperDistanceRaw-1000.0f-paperDistanceOffset[0])/(1000.0f+paperDistanceOffset[10]-paperDistanceOffset[0]);
            // 限幅
            if(paperDistanceNormalization >= 1.0f)
                paperDistanceNormalization = 1.0f;
            else if(paperDistanceNormalization <= 0.0f)
                paperDistanceNormalization = 0.0f;
            int idx = 0;    // 偏移数组索引
            for(int i = 0;i < (1000/OFFSET_STEP_DISTANCE);i++){
                if(1.0/(1000.0/float(OFFSET_STEP_DISTANCE))*(float)i <= paperDistanceNormalization && paperDistanceNormalization <= 1/(1000/OFFSET_STEP_DISTANCE)*(float)(i+1)){
                    idx = i;
                    break;
                }
            }
            paperDistanceCalibration = paperDistanceRaw-paperDistanceOffset[idx]-((1000.0/float(OFFSET_STEP_DISTANCE))*paperDistanceNormalization-(float)idx)*(paperDistanceOffset[idx+1]-paperDistanceOffset[idx]); // 距离修正
            // 将旋转向量转换为旋转矩阵
            Mat rotMat;
            Rodrigues(rvec, rotMat);
            // 计算欧拉角
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            paperPitch = atan2(R.at<double>(2, 1), R.at<double>(2, 2)) * 180 / CV_PI;
            paperYaw = atan2(-R.at<double>(2, 0), sqrt(pow(R.at<double>(2, 1), 2) + pow(R.at<double>(2, 2), 2))) * 180 / CV_PI;
            paperRoll = atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * 180 / CV_PI;
        }
    }
}

// 绘制三维坐标轴
void drawCoordinateAxes(Mat& image, const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec, float length) {
    vector<Point3f> axesPoints = {
        {0, 0, 0},          // 坐标系原点
        {length, 0, 0},     // X轴
        {0, length, 0},     // Y轴
        {0, 0, length}      // Z轴
    };

    std::vector<cv::Point2f> projectedPoints;

    if (quadsOrderPoints.size() == 4)
        projectPoints(axesPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
    
    // 绘制坐标轴
    if(projectedPoints.size() == 4){
        line(image, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 4); // X轴（红色）
        line(image, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 4); // Y轴（绿色）
        line(image, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 4); // Z轴（蓝色）
    }
}

// 内部形状获取（0：三角形；1：正方形；2：圆形）
void innerShapeGet(vector<vector<Point>> Contours, bool _isOverlap_){
    // 初始化
    innerShape.clear();
    innerShapeEdgeLength.clear();
    innerContours.clear();
    innerHierarchy.clear();
    innerRect.clear();
    innerPoints.clear();
    aiDetectionNum = 10;   
    aiDetectionROI.clear(); 
    aiDetectionNum2Idx.clear();
    if(Contours.size() != 0){
        // // 设置ROI掩膜
        Mat mask = Mat::zeros(ImgColor.size(), CV_8UC1);
        drawContours(mask, Contours, 0, Scalar(255), FILLED);
        // 复制图像到掩膜
        Mat ImgROIGray;
        Mat ImgROIBinary;
        Mat ImgROIEdge;
        ImgROIGray = ImgColor.clone();
        JetsonImage.colorConvert(ImgROIGray,COLOR_BGR2GRAY);
        ImgROIBinary = ImgROIGray.clone();
        JetsonImageBGR.thresholdConvert(ImgROIBinary);
        bitwise_not(ImgROIBinary, ImgROIBinary);
        // 边缘提取
        Canny(ImgROIGray, ImgROIEdge, 100, 150, 3);
        ImgROI.release();
        ImgROIBinary.copyTo(ImgROI, mask);
        // 将轮廓点上的噪点遮盖
        for(unsigned long int i = 0;i < Contours[0].size();i++){
            circle(ImgROI, Contours[0][i], 1, cv::Scalar(0), -1);
        }
        // bitwise_not(ImgROI, ImgROI);

        // 查找内部轮廓
        JetsonImage.findContours(ImgROI, innerContours, innerHierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);      // 存储所有点
        // 找内部轮廓最小外接矩形
        JetsonImage.findContoursRect(innerContours,innerRect);
        // 绘制内部轮廓
        JetsonImage.drawContours(ImgTest, innerContours, innerHierarchy, Scalar(255,0,255));
        // 绘制内部轮廓最小外接矩形
        JetsonImage.drawContoursRect(ImgTest, innerRect, Scalar(0,255,255));
        
        for(unsigned long int i = 0;i < innerContours.size();i++){
            // cout << innerContours.size() << endl;
            // cout << contourArea(innerContours[i]) << endl;
            // 大于最小图形轮廓面积才能判断形状/在外侧/无重叠情况
            if(contourArea(innerContours[i]) > INNER_SHAPE_SQUARE_MIN && isOverlap == false){         
                // 计算轮廓周长
                double perimeter = arcLength(innerContours[i], true);
                // 多边形逼近（epsilon为周长的一定比例）
                vector<Point2f> innerPointsBuf;    // 内部图形轮廓点
debugPoint(71);
                approxPolyDP(innerContours[i], innerPointsBuf, 0.04 * perimeter, true);
                innerPoints.push_back(innerPointsBuf);
                
                Point2f center;
                float radius;
                double squareRate;  // 轮廓面积和最小外接圆面积比值
                // 找最小外接圆
                minEnclosingCircle(innerContours[i], center, radius);
                // 计算轮廓面积
                double squareContour = contourArea(innerContours[i]);
                // 计算比值
                squareRate = squareContour / (CV_PI * radius * radius);

                // 形状判断
                // 三角形
                if(innerPointsBuf.size() == 3){
                    if(squareRate < 0.85){  // 若轮廓面积/最小外接圆面积小于阈值 -> 不是圆形
                        innerShape.push_back(0);
                        // 边长/半径计算
                        innerShapeEdgeLength.push_back(perimeter/3.0f);
                    }
                    else{
                        innerShape.push_back(2);
                        innerShapeEdgeLength.push_back(perimeter/CV_PI);
                    }
                }
                // 正方形
                else if(innerPointsBuf.size() == 4){
                    if(squareRate < 0.85){      // 若轮廓面积/最小外接圆面积小于阈值 -> 不是圆形
                        double squareRate;  // 轮廓面积和最小外接矩形面积比值
                        // 计算轮廓面积
                        double squareContour = contourArea(innerContours[i]);
                        //计算比值
                        squareRate = squareContour / (innerRect[i].size.height*innerRect[i].size.width);
                        // 比值大于阈值，判定为矩形
                        if(squareRate > 0.85){
                            innerShape.push_back(1);
                            // 边长/半径计算
                            innerShapeEdgeLength.push_back(perimeter/4.0f);
                        }
                        else{
                            isOverlap = true;
                            innerShapeGet(filterContoursNMS, isOverlap);
                        }
                    }
                    else{
                        innerShape.push_back(2);
                        innerShapeEdgeLength.push_back(perimeter/CV_PI);
                    }
                }
                // 圆形
                else if(innerPointsBuf.size() > 4){
                    // 比值大于阈值，判定为圆形
                    if(squareRate > 0.85){
                        innerShape.push_back(2);
                        innerShapeEdgeLength.push_back(perimeter/CV_PI);
                    }
                    else{
                        isOverlap = true;
                        innerShapeGet(filterContoursNMS, isOverlap);
                    }
                }
            }
            // 重叠
            else if(contourArea(innerContours[i]) > INNER_SHAPE_SQUARE_MIN && isOverlap == true){
                // 计算轮廓周长
                double perimeter = arcLength(innerContours[i], true);
                // 多边形逼近（epsilon为周长的一定比例）
                vector<Point2f> innerPointsBuf;    // 内部图形轮廓点
debugPoint(72);
                approxPolyDP(innerContours[i], innerPointsBuf, 0.008 * perimeter, true);
                innerPoints.push_back(innerPointsBuf);

                vector<bool> innerPointKind;
                // 点分类，分为边框角点和相交点
                for (unsigned int i = 0; i < innerPointsBuf.size(); ++i) {
                    Point point = Point(innerPointsBuf[i].x,innerPointsBuf[i].y);
                    // 创建圆形掩模
                    Mat mask = Mat::zeros(ImgGray.size(), CV_8UC1);
                    circle(mask, point, POINT_SORT_CIRCLE_ROI_RADIUS, 255, -1);
                    // 提取圆形ROI
                    Mat ImgCircleROIGray;
                    ImgGray.copyTo(ImgCircleROIGray, mask);
                    // 二值化
                    Mat ImgCircleROIBinary = ImgCircleROIGray.clone();
                    JetsonImageBGR.thresholdConvert(ImgCircleROIBinary);
                    // 像素统计
                    int whitePixels = cv::countNonZero(ImgCircleROIBinary);
                    int totalPixels = cv::countNonZero(mask);
                    // 计算黑色像素比例
                    float blackRatio = 1.0f - (whitePixels / static_cast<float>(totalPixels));
                    blackRatio = round(blackRatio * 100) / 100; // 保留两位小数
                    // 点类型分类
                    if (blackRatio <= 0.35) {
                        innerPointKind.push_back(true);
                    } else {
                        innerPointKind.push_back(false);
                    }
                }
                overlapRectExtract(innerPointsBuf, innerPointKind);
                innerPointsKind.push_back(innerPointKind);  // 存储点类型
            }
        }
    }
}

// 重叠矩形提取
void overlapRectExtract(vector<Point2f> innerPoint, vector<bool> innerPointKind){
    // 初始化
    innerShape.clear();
    innerContours.clear();
    innerHierarchy.clear();
    innerRect.clear();
    innerSplitPoints.clear();
    innerSplitTruePoints.clear();
    innerSplitPointsKind.clear();
    innerShapeEdgeLength.clear();

    if(innerPoint.size() == innerPointKind.size()){
        // 使相交点排第一个
        for(unsigned long int i = 0;i < innerPoint.size();i++){
            if(innerPointKind[i] == false){
                // 旋转点集，使相交点在最前，顺时针排序
                std::rotate(innerPoint.begin(), innerPoint.begin()+i, innerPoint.end());
                std::rotate(innerPointKind.begin(), innerPointKind.begin()+i, innerPointKind.end());
            }
        }
        vector<Point2f> innerRectIntersectionPoint;     // 内部图形矩形交点
        int truePointNum = 0;   // 边框角点数量
        // 判断矩形上交点是否属于同一个矩形
        for(unsigned long int i = 0;i <= innerPoint.size();i++){
            if(innerPointKind[i%innerPoint.size()] == false){     // 矩形交点
                innerRectIntersectionPoint.push_back(innerPoint[i%innerPoint.size()]);
                if(truePointNum == 3){     // 相隔三个边框角点
                    line(ImgROI,innerRectIntersectionPoint[innerRectIntersectionPoint.size()-1],innerRectIntersectionPoint[innerRectIntersectionPoint.size()-2],Scalar(0),2);   // 重合矩形分割
                }
                truePointNum = 0;
            }
            else{   // 矩形边框角点
                truePointNum++;
            }
        }
        // 边缘提取
        // Canny(ImgROI, ImgROI, 100, 150, 3);
        vector<vector<Point>> innerSplitContours;    // 内部分割图形轮廓
        vector<RotatedRect> innerSplitRect;          // 内部分割图形轮廓最小外接矩形
        vector<Vec4i> innerSplitHierarchy;           // 内部分割图形索引
        // 查找分割后的重合矩形轮廓
        JetsonImage.findContours(ImgROI, innerSplitContours, innerSplitHierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);      // 存储所有点
        // 遍历所有分割轮廓
        for(unsigned long int i = 0;i < innerSplitContours.size();i++){
            // 计算分割后的重合矩形轮廓周长
            double perimeter = arcLength(innerSplitContours[i], true);
            // 多边形逼近（epsilon为周长的一定比例）
            vector<Point2f> innerSplitPointsBuf;    // 内部分割图形轮廓点
            approxPolyDP(innerSplitContours[i], innerSplitPointsBuf, 0.02 * perimeter, true);
            innerSplitPoints.push_back(innerSplitPointsBuf);

            vector<bool> innerSplitPointKind;
            vector<Point2f> innerSplitTruePointsBuf;    // 内部分割图形轮廓边框角点
            int trueSplitPointNum = 0;  // 分割图形边框角点数量
            // 点分类，分为边框角点和相交点
            for (unsigned int i = 0; i < innerSplitPointsBuf.size(); ++i) {
                Point point = Point(innerSplitPointsBuf[i].x,innerSplitPointsBuf[i].y);
                // 创建圆形掩模
                Mat mask = Mat::zeros(ImgGray.size(), CV_8UC1);
                circle(mask, point, POINT_SORT_CIRCLE_ROI_RADIUS, 255, -1);
                // 提取圆形ROI
                Mat ImgCircleROIGray;
                ImgGray.copyTo(ImgCircleROIGray, mask);
                // 二值化
                Mat ImgCircleROIBinary = ImgCircleROIGray.clone();
                JetsonImageBGR.thresholdConvert(ImgCircleROIBinary);
                // 像素统计
                int whitePixels = cv::countNonZero(ImgCircleROIBinary);
                int totalPixels = cv::countNonZero(mask);
                // 计算黑色像素比例
                float blackRatio = 1.0f - (whitePixels / static_cast<float>(totalPixels));
                blackRatio = round(blackRatio * 100) / 100; // 保留两位小数
                // 点类型分类
                if (blackRatio <= 0.30) {
                    innerSplitPointKind.push_back(true);
                    innerSplitTruePointsBuf.push_back(innerSplitPointsBuf[i]);
                    trueSplitPointNum++;
                } else {
                    innerSplitPointKind.push_back(false);
                }
            }
            // 计算边长
            // 分割图形边框角点数量为2时，该两点连线为对角线；分割图形边框角点数量为3时，该三点连线为边框线      
            if(trueSplitPointNum == 2){
                innerShape.push_back(1);
                innerShapeEdgeLength.push_back(norm(innerSplitTruePointsBuf[0]-innerSplitTruePointsBuf[1])/1.4142135623730950488016887242);
                // 绘制边框角点
                for(unsigned long int i = 0;i < innerSplitTruePointsBuf.size();i++){
                    circle(ImgTest, innerSplitTruePointsBuf[i], 3, cv::Scalar(0, 0, 255), -1);
                }
                // 画正方形
                from2PointDrawRotatedSquare(ImgTest, innerSplitTruePointsBuf[0], innerSplitTruePointsBuf[1], Scalar(255,0,0));
            }
            else if(trueSplitPointNum == 3){
                double maxDistance = 0;
                Point2f p1;     // 对角点1
                Point2f p2;     // 对角点2
                // 绘制边框角点
                for(unsigned long int i = 0;i < innerSplitTruePointsBuf.size();i++){
                    circle(ImgTest, innerSplitTruePointsBuf[i], 3, cv::Scalar(0, 0, 255), -1);
                }
                // 计算对角点
                for(unsigned long int i = 0;i < innerSplitTruePointsBuf.size();i++){
                    if(norm(innerSplitTruePointsBuf[i]-innerSplitTruePointsBuf[(i+1)%3]) >= maxDistance){
                        p1 = innerSplitTruePointsBuf[i];
                        p2 = innerSplitTruePointsBuf[(i+1)%3];
                        maxDistance = norm(innerSplitTruePointsBuf[i]-innerSplitTruePointsBuf[(i+1)%3]);
                    }
                }
                innerShape.push_back(1);
                innerShapeEdgeLength.push_back(norm(p1-p2)/1.4142135623730950488016887242);
                from2PointDrawRotatedSquare(ImgTest, p1, p2, Scalar(255,0,0));
            }
            innerSplitPointsKind.push_back(innerSplitPointKind);
            innerSplitTruePoints.push_back(innerSplitTruePointsBuf);
        }
        JetsonImage.drawContours(ImgTest, innerSplitContours, innerHierarchy, Scalar(0,255,0));
        // cout << innerSplitContours.size() << endl;
    }
}

// 正方形绘制(对角线点)
void from2PointDrawRotatedSquare(Mat& image, Point2f pt1, Point2f pt2, Scalar color){
    // 计算中心点
    Point2f center((pt1.x + pt2.x) / 2.0f, (pt1.y + pt2.y) / 2.0f);
    
    // 计算对角线长度作为正方形对角线
    double diagonal = norm(pt1 - pt2);
    double side = diagonal / sqrt(2);  // 正方形边长
    
    // 计算旋转角度（以pt1到pt2的方向为基准）
    double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180 / CV_PI;
    
    // 创建旋转矩形
    RotatedRect rotatedRect(center, Size2f(side, side), angle-45);
    
    // 获取四个顶点
    Point2f vertices[4];
    vector<Point2f> aiRoiPoint;
    rotatedRect.points(vertices);

    for(int i = 0;i < 4;i++)
        aiRoiPoint.push_back(vertices[i]);

    // 添加AI识别ROI
    aiDetectionROI.push_back(boundingRect(aiRoiPoint));
    
    // 绘制正方形
    for (int i = 0; i < 4; i++) {
        line(image, vertices[i], vertices[(i + 1) % 4], color, 1);
    }
}

// AI识别
void AI(void){
    // 遍历所有AI ROI
    for(unsigned long int i = 0;i < aiDetectionROI.size();i++){
        int result;
        // ai代码

        aiDetectionNum = result;
        aiDetectionNum2Idx.push_back({aiDetectionNum, i});
    }
}

// 求最小值


// DEBUG软件打点
void debugPoint(int step){
    #if DEBUG_EN == 1
    cout << "STEP: " << step << endl;
    #endif
}