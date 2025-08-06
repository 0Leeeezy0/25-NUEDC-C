#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <dirent.h>  // 替代filesystem的目录操作

using namespace cv;
using namespace std;

// 全局变量
Size boardSize(8, 6);        // 棋盘格内角点数量 (宽,高)
float squareSize = 24.0f;     // 每个方格的实际尺寸(mm)
vector<vector<Point2f>> imagePoints;  // 存储所有图像的角点
vector<vector<Point3f>> objectPoints; // 存储所有3D点
Mat cameraMatrix, distCoeffs;        // 相机内参和畸变系数
Size imageSize;                     // 图像尺寸
int imagesCount = 0;          // 已采集的有效图像数
const int minImages = 30;      // 最少需要采集的图像数

// 初始化3D对象点
vector<Point3f> initObjectPoints() {
    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            obj.push_back(Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    return obj;
}

// 检测棋盘格角点
bool detectChessboardCorners(Mat &image, vector<Point2f> &corners) {
    bool found = findChessboardCorners(image, boardSize, corners, 
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
    
    if (found) {
        // 亚像素精确化
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);
    }
    return found;
}

// 执行相机标定
void performCameraCalibration() {
    if (imagePoints.size() < minImages) {
        cout << "需要至少 " << minImages << " 张有效图像，当前只有 " << imagePoints.size() << " 张" << endl;
        return;
    }

    // 初始化对象点（所有图像使用相同的3D点）
    vector<vector<Point3f>> objPoints(imagePoints.size(), initObjectPoints());

    // 标定相机
    vector<Mat> rvecs, tvecs;
    int flags = CALIB_RATIONAL_MODEL | CALIB_FIX_K3;
    double rms = ::calibrateCamera(objPoints, imagePoints, imageSize, 
                                 cameraMatrix, distCoeffs, rvecs, tvecs, flags);

    cout << "\n标定结果:" << endl;
    cout << "RMS误差: " << rms << " 像素" << endl;
    cout << "相机内参矩阵:\n" << cameraMatrix << endl;
    cout << "畸变系数:\n" << distCoeffs.t() << endl;

    // 保存标定结果
    FileStorage fs("camera_calibration.yml", FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "rms_error" << rms;
    fs.release();
    cout << "标定结果已保存到 camera_calibration.yml" << endl;
}

// 主函数
int main() {
    VideoCapture cap;  // 打开默认相机
    cap.open("/dev/video0",CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 640); // 设置摄像头横向分辨率
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);  // 设置摄像头纵向分辨率
    if (!cap.isOpened()) {
        cerr << "无法打开相机!" << endl;
        return -1;
    }

    // 获取第一帧以确定图像尺寸
    Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) {
        cerr << "无法获取相机帧!" << endl;
        return -1;
    }
    imageSize = firstFrame.size();

    cout << "相机标定程序" << endl;
    cout << "使用方法:" << endl;
    cout << "1. 准备 " << boardSize.width-1 << "x" << boardSize.height-1 << " 的棋盘格" << endl;
    cout << "2. 按 'c' 键捕获当前帧" << endl;
    cout << "3. 捕获至少 " << minImages << " 张不同角度的图像后按 'q' 开始标定" << endl;

    Mat frame;
    namedWindow("Camera Calibration", WINDOW_NORMAL);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 尝试检测棋盘格
        vector<Point2f> corners;
        bool found = detectChessboardCorners(frame, corners);

        // 显示结果
        Mat view = frame.clone();
        if (found) {
            drawChessboardCorners(view, boardSize, Mat(corners), found);
            putText(view, format("GET: %d/%d", imagesCount, minImages), 
                    Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        } else {
            putText(view, "NO FOUND", Point(20, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }

        imshow("Camera Calibration", view);

        // 键盘交互
        char key = waitKey(10);
        if (key == 'q') break;
        if (key == 'c' && found) {
            imagePoints.push_back(corners);
            imagesCount++;
            cout << "已捕获图像 " << imagesCount << endl;
            
            // 短暂显示绿色边框表示捕获成功
            rectangle(view, Point(0, 0), Point(view.cols-1, view.rows-1), Scalar(0, 255, 0), 5);
            imshow("Camera Calibration", view);
            waitKey(300);
        }
    }

    // 执行标定
    if (!imagePoints.empty()) {
        performCameraCalibration();
        
        // 显示标定后的效果
        Mat map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), 
                              getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1), 
                              imageSize, CV_16SC2, map1, map2);
        
        cout << "\n按任意键查看去畸变效果，按'q'退出..." << endl;
        while (true) {
            Mat frame, undistorted;
            cap >> frame;
            if (frame.empty()) break;
            
            remap(frame, undistorted, map1, map2, INTER_LINEAR);
            imshow("原始图像", frame);
            imshow("去畸变图像", undistorted);
            
            if (waitKey(30) >= 0) break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}