from maix import image, camera, display, uart, pinmap, nn
import cv2
import numpy as np
import time
import math
import os
from typing import List, Tuple, Optional
#---------------------------------------------------------------------------------------------------------#
detector = nn.YOLOv5(model="/root/models/7.14/best5.mud",dual_buff=False)
#---------------------------------------------------------------------------------------------------------#
# 全局变量
CHOOSE_CONTOUR_IDX = 1                  # 选择非极大值抑制轮廓的索引（若外框在最外侧：1，若外框在内测：0）
SQUARE_MIN = 5000.0                     # 面积筛选最小值
SQUARE_MAX = 110000.0                   # 面积筛选最大值
IMG_WIDTH = 640                         # 图像宽度
IMG_HEIGHT = 480                        # 图像高度
CIRCLE_ROI_RADIUS = 80                  # 圆形ROI区域半径（坐标筛选ROI）
OFFSET_STEP_DISTANCE = 100              # 偏移标定步长（mm）
INNER_SHAPE_SQUARE_MIN = 800.0         # 内部图形面积最小值
CALIBRATION_WIDTH = 168.1               # A4宽度校准真实值（mm）
CALIBRATION_HEIGHT = 255.1             # A4高度校准真实值（mm）
POINT_SORT_CIRCLE_ROI_RADIUS = 10        # 圆形ROI区域半径（点分类ROI）
POINT_SORT_LIMIT = 0.30                  # 点分类比例阈值（黑像素/全部圆周像素）
DILATE_CORE_SIZE = 8                    # 膨胀核大小（黑色变少）
ERODE_CORE_SIZE = 8                     # 腐蚀核大小（黑色变多）

DEBUG_EN = False                        # 调试使能
PRINT_EN = True                         # 打印使能
UART_EN = True                          # 串口使能
GAME_EN = True                         # 比赛使能
# 相机参数
# 相机内参矩阵
cameraMatrix = np.array([
    [1158.86187, 0, 357.724676],
    [0, 1183.95199, 200.274766],
    [0, 0, 1]
], dtype=np.float64)

# 相机畸变系数
distCoeffs = np.array([-0.105515381, 44.0325107, 0.00264077449, -0.000451745137, 0, 0.273236799, 45.8046971, -0.151397028], dtype=np.float64)

# 1000mm-2000mm处偏移，步进值 OFFSET_STEP_DISTANCE（mm）
# paperDistanceOffset = [190.00, 205.50, 216.50, 241.50, 265.00, 278.01, 291.90, 311.50 ,332.53, 343.00, 397.00]#//2
paperDistanceOffset = [0,0,0,0,0,0,0,0,0,0,0]#//2

# 真实A4纸坐标
objectPoints = np.array([
    [-CALIBRATION_WIDTH/2, -CALIBRATION_HEIGHT/2, 0],
    [CALIBRATION_WIDTH/2, -CALIBRATION_HEIGHT/2, 0],
    [CALIBRATION_WIDTH/2, CALIBRATION_HEIGHT/2, 0],
    [-CALIBRATION_WIDTH/2, CALIBRATION_HEIGHT/2, 0]
], dtype=np.float32)

# 全局变量（原C++中的全局变量）
ImgColor = []
ImgColorTrans = []
ImgGray = []
ImgGrayTrans = []
ImgBinary = []
ImgBinaryTrans = []
ImgEdge = []
ImgROI = []
ImgNano = []
ImgTest = []
allContours = []
allRect = []
allCenter = []
allHierarchy = []
filterContours = []
filterRect = []
filterCenter = []
filterContoursNMS = []
filterRectNMS = []
filterCenterNMS = []
quadsPoints = []
quadsOrderPoints = []
quadsEdgeLength = []
quadsEdgePerimeter = 0.0
innerContours = []
innerHierarchy = []
innerPoints = []
innerPointsKind = []
innerSplitContours = []
innerSplitHierarchy = []
innerSplitPoints = []
innerSplitTruePoints = []
innerSplitPointsKind = []
paperDistanceRaw = 0.0
paperDistanceCalibration = 0.0
paperYaw = 0.0
paperPitch = 0.0
paperRoll = 0.0
isOverlap = False
rvec = None
tvec = None
innerShape = []
innerShapeEdgeLength = []
neededNum = 10
aiDetectionNum = 10
aiDetectionROI = []
aiDetectionNum2Idx = []

# RGB通道
class BGR_Channel:
    B_Channel = 0
    G_Channel = 1
    R_Channel = 2

class Image:
    def __init__(self):
        self.ImgSynthesis = []  # 用于存储待合成的图像
    
    class BGR:
        @staticmethod
        def thresholdConvert(img: np.ndarray) -> np.ndarray:
            """灰度图像二值化（使用OTSU算法）"""
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return thresh
        
        @staticmethod
        def channelSplit(img: np.ndarray, channel: int) -> np.ndarray:
            """BGR图像通道提取"""
            channels = cv2.split(img)
            # 将其他通道置零
            if channel == BGRChannel.B_Channel:
                channels[1] = np.zeros_like(channels[1])
                channels[2] = np.zeros_like(channels[2])
            elif channel == BGRChannel.G_Channel:
                channels[0] = np.zeros_like(channels[0])
                channels[2] = np.zeros_like(channels[2])
            elif channel == BGRChannel.R_Channel:
                channels[0] = np.zeros_like(channels[0])
                channels[1] = np.zeros_like(channels[1])
            return cv2.merge(channels)
    
    class HSV:
        @staticmethod
        def Equalize(img: np.ndarray) -> np.ndarray:
            """HSV图像亮度通道直方图均衡化"""
            hsv = cv2.split(img)
            hsv[2] = cv2.equalizeHist(hsv[2])  # 对V通道均衡化
            return cv2.merge(hsv)
        
        @staticmethod
        def thresholdConvert(img: np.ndarray, low_limit: Tuple[int, int, int], high_limit: Tuple[int, int, int]) -> np.ndarray:
            """HSV图像二值化（多阈值）"""
            return cv2.inRange(img, low_limit, high_limit)
    
    @staticmethod
    def colorConvert(img: np.ndarray, code: int) -> np.ndarray:
        """图像色彩空间转换"""
        return cv2.cvtColor(img, code)
    
    @staticmethod
    def Sobel(img: np.ndarray, sobel_core: int) -> np.ndarray:
        """Sobel算子边缘检测"""
        # X方向边缘检测
        img_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=sobel_core)
        img_x = cv2.convertScaleAbs(img_x)
        
        # Y方向边缘检测
        img_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=sobel_core)
        img_y = cv2.convertScaleAbs(img_y)
        
        # 图像混合
        return cv2.addWeighted(img_x, 0.5, img_y, 0.5, 0)
    
    @staticmethod
    def Scharr(img: np.ndarray, scharr_core: int) -> np.ndarray:
        """Scharr算子边缘检测"""
        # X方向边缘检测
        img_x = cv2.Scharr(img, cv2.CV_16S, 1, 0, scale=scharr_core)
        img_x = cv2.convertScaleAbs(img_x)
        
        # Y方向边缘检测
        img_y = cv2.Scharr(img, cv2.CV_16S, 0, 1, scale=scharr_core)
        img_y = cv2.convertScaleAbs(img_y)
        
        # 图像混合
        return cv2.addWeighted(img_x, 0.5, img_y, 0.5, 0)
    
    @staticmethod
    def Canny(img: np.ndarray, low: int, high: int, size: int) -> np.ndarray:
        """Canny算子边缘检测"""
        return cv2.Canny(img, low, high, apertureSize=size, L2gradient=False)
    
    @staticmethod
    def GaussBlur(img: np.ndarray, blur_size: int) -> np.ndarray:
        """高斯模糊"""
        return cv2.GaussianBlur(img, (blur_size, blur_size), 3, 3)
    
    @staticmethod
    def Sharpen(img: np.ndarray, blur_size: int) -> np.ndarray:
        """图像锐化"""
        img_gauss = cv2.GaussianBlur(img, (blur_size, blur_size), 3, 3)
        return cv2.addWeighted(img, 2, img_gauss, -1, 0)
    
    @staticmethod
    def Dilate(img: np.ndarray, core_size: int) -> np.ndarray:
        """形态学膨胀"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (core_size, core_size))
        return cv2.dilate(img, kernel)
    
    @staticmethod
    def Erode(img: np.ndarray, core_size: int) -> np.ndarray:
        """形态学腐蚀"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (core_size, core_size))
        return cv2.erode(img, kernel)
    
    @staticmethod
    def Unpivot(img: np.ndarray, src_points: List[Tuple[float, float]], dst_points: List[Tuple[float, float]]) -> np.ndarray:
        """透视变换"""
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, (320, 240), flags=cv2.INTER_LINEAR)

    @staticmethod
    def findContours(img: np.ndarray, mode: int, method: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """寻找轮廓"""
        contours, hierarchy = cv2.findContours(img, mode, method, offset=(0, 0))
        return contours, hierarchy
    
    @staticmethod
    def drawContours(img: np.ndarray, contours: List[np.ndarray], hierarchy: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """绘制轮廓"""
        for i in range(len(contours)):
            cv2.drawContours(img, contours, i, color, 2, 8, hierarchy, 0)
        return img
    
    @staticmethod
    def findContoursRect(contours: List[np.ndarray]) -> List[cv2.RotatedRect]:
        """获取轮廓的最小外接矩形"""
        return [cv2.minAreaRect(contour) for contour in contours]
    
    @staticmethod
    def findContoursCenter(rects: List[cv2.RotatedRect]) -> List[Tuple[int, int]]:
        """获取轮廓外接矩形的中心点"""
        return [tuple(map(int, rect[0])) for rect in rects]
    
    @staticmethod
    def drawContoursRect(img: np.ndarray, rects: List[cv2.RotatedRect], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """绘制轮廓的外接矩形"""
        for rect in rects:
            # 获取矩形的四个顶点
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, color, 2)
        return img

    @staticmethod
    def newImg(img: np.ndarray, width: int, height: int, img_type: int, color: Tuple[int, int, int]) -> None:
        """新建画布"""
        img[:] = np.full((height, width, 3), color, dtype=np.uint8) if img_type == cv2.CV_8UC3 else np.zeros((height, width), dtype=np.uint8)

# 25-NUEDC-C
def initVariable() -> None:
    """变量初始化"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx
    
    allContours = []
    allRect = []
    allCenter = []
    allHierarchy = []
    filterContours = []
    filterRect = []
    filterCenter = []
    filterContoursNMS = []
    filterRectNMS = []
    filterCenterNMS = []
    quadsPoints = []
    quadsOrderPoints = []
    quadsEdgeLength = []
    quadsEdgePerimeter = 0.0
    innerContours = []
    innerHierarchy = []
    innerPoints = []
    innerPointsKind = []
    innerSplitContours = []
    innerSplitHierarchy = []
    innerSplitPoints = []
    innerSplitTruePoints = []
    innerSplitPointsKind = []
    paperDistanceRaw = 0.0
    paperDistanceCalibration = 0.0
    paperYaw = 0.0
    paperPitch = 0.0
    paperRoll = 0.0
    isOverlap = False
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    innerShape = []
    innerShapeEdgeLength = []
    neededNum = 10
    aiDetectionNum = 10
    aiDetectionROI = []
    aiDetectionNum2Idx = []


def imgProcess() -> None:
    """图像处理"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx
    
    img = cam.read()
    frame = image.image2cv(img)
    ImgColor = frame.copy()
    
    # 新建画布
    ImgROI = np.zeros((HEIGHT_480, WIDTH_640), dtype=np.uint8)
    GE_Image.newImg(ImgROI, WIDTH_640, HEIGHT_480, cv2.CV_8UC1, (0, 0, 0))
    ImgNano = ImgColor.copy()
    ImgTest = ImgColor.copy()
    ImgColorTrans = ImgColor.copy()
    ImgGray = GE_Image.colorConvert(ImgColor, cv2.COLOR_BGR2GRAY)
    ImgGrayTrans = ImgGray.copy()

    # 图像腐蚀膨胀
    ImgGrayClosing = GE_Image.Dilate(ImgGray, DILATE_CORE_SIZE)   # 黑色变少
    ImgGrayClosing = GE_Image.Erode(ImgGrayClosing, ERODE_CORE_SIZE)   # 黑色变多
    ImgBinary = GE_Image.BGR.thresholdConvert(ImgGrayClosing)
    ImgBinaryTrans = ImgBinary.copy()

    # Canny边缘检测
    ImgEdge = cv2.Canny(ImgGrayClosing, 100, 150, 3)
    
    # 绘制参考线
    cv2.line(ImgNano, (0, IMG_HEIGHT//2), (IMG_WIDTH, IMG_HEIGHT//2), (255, 255, 0), 2)
    cv2.line(ImgNano, (IMG_WIDTH//2, 0), (IMG_WIDTH//2, IMG_HEIGHT), (255, 255, 0), 2)
    cv2.circle(ImgNano, (IMG_WIDTH//2, IMG_HEIGHT//2), CIRCLE_ROI_RADIUS, (255, 255, 0), 2)


def contourFilter(contours: List[np.ndarray]) -> None:
    """轮廓筛选"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    square_filter_contours = []
    square_filter_rect = []
    square_filter_center = []
    
    filterContours = []
    filterRect = []
    filterCenter = []
    
    if allContours:
        allRect = GE_Image.findContoursRect(allContours)
        allCenter = GE_Image.findContoursCenter(allRect)
    
    # 面积筛选
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if SQUARE_MIN < contour_area < SQUARE_MAX and len(contour) > 0:
            square_filter_contours.append(contour)
    
    if square_filter_contours:
        square_filter_rect = GE_Image.findContoursRect(square_filter_contours)
        square_filter_center = GE_Image.findContoursCenter(square_filter_rect)
    
    # 坐标筛选
    for i in range(len(square_filter_rect)):
        center = square_filter_center[i]
        dist = math.hypot(center[0] - IMG_WIDTH/2, center[1] - IMG_HEIGHT/2)
        if dist < CIRCLE_ROI_RADIUS:
            filterContours.append(square_filter_contours[i])
            filterRect.append(square_filter_rect[i])
            filterCenter.append(center)


def contourNMS(contours: List[np.ndarray]) -> None:
    """非极大值抑制"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    filterContoursNMS = []
    filterRectNMS = []
    filterCenterNMS = []
    
    if contours:
        areas = [(cv2.contourArea(c), i) for i, c in enumerate(contours)]
        areas.sort(reverse=True, key=lambda x: x[0])
        
        sorted_contours = [contours[i] for (area, i) in areas]
        
        if len(sorted_contours) > CHOOSE_CONTOUR_IDX:
            filterContoursNMS.append(sorted_contours[CHOOSE_CONTOUR_IDX])
        elif len(sorted_contours) == 1:
            filterContoursNMS.append(sorted_contours[0])
        
        if filterContoursNMS:
            filterRectNMS = GE_Image.findContoursRect(filterContoursNMS)
            filterCenterNMS = GE_Image.findContoursCenter(filterRectNMS)


def Contours2Quads(contours: List[np.ndarray], center: List[Tuple[int, int]]) -> None:
    """轮廓四边形拟合"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    quadsOrderPoints = []
    quadsEdgeLength = []
    
    if contours:
        # 计算轮廓周长
        quadsEdgePerimeter = cv2.arcLength(contours[0], True)
        # 多边形逼近
        quadsPoints = cv2.approxPolyDP(contours[0], 0.02 * quadsEdgePerimeter, True)
        
        if len(quadsPoints) == 4 and center:
            # 角点顺序整定
            center_pt = center[0]
            # 左上
            for p in quadsPoints:
                pt = (p[0][0], p[0][1])
                if pt[0] < center_pt[0] and pt[1] < center_pt[1]:
                    quadsOrderPoints.append(pt)
            # 右上
            for p in quadsPoints:
                pt = (p[0][0], p[0][1])
                if pt[0] > center_pt[0] and pt[1] < center_pt[1]:
                    quadsOrderPoints.append(pt)
            # 右下
            for p in quadsPoints:
                pt = (p[0][0], p[0][1])
                if pt[0] > center_pt[0] and pt[1] > center_pt[1]:
                    quadsOrderPoints.append(pt)
            # 左下
            for p in quadsPoints:
                pt = (p[0][0], p[0][1])
                if pt[0] < center_pt[0] and pt[1] > center_pt[1]:
                    quadsOrderPoints.append(pt)
            
            # 计算边长
            if len(quadsOrderPoints) == 4:
                for i in range(4):
                    x1, y1 = quadsOrderPoints[i]
                    x2, y2 = quadsOrderPoints[(i+1)%4]
                    distance = math.hypot(x2 - x1, y2 - y1)
                    quadsEdgeLength.append(distance)


def calu4DistanceAndEularAngle(camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
    """距离和欧拉角计算"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    if len(quadsOrderPoints) == 4:
        # 定义图像坐标
        image_points = np.array(quadsOrderPoints, dtype=np.float32)
        
        _, rvec, tvec = cv2.solvePnP(objectPoints, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
        
        if rvec is not None and tvec is not None:
            paperDistanceRaw = tvec[2][0]
            # 距离归一化
            paper_distance_normalization = (paperDistanceRaw - 1000.0 - paperDistanceOffset[0]) / \
                                         (1000.0 + paperDistanceOffset[10] - paperDistanceOffset[0])
            paper_distance_normalization = max(0.0, min(1.0, paper_distance_normalization))
            
            idx = 0
            for i in range(1000 // OFFSET_STEP_DISTANCE):
                if (1.0 / (1000.0 / OFFSET_STEP_DISTANCE)) * i <= paper_distance_normalization <= \
                   (1.0 / (1000.0 / OFFSET_STEP_DISTANCE)) * (i + 1):
                    idx = i
                    break
            
            paperDistanceCalibration = paperDistanceRaw - paperDistanceOffset[idx] - \
                                     ((1000.0 / OFFSET_STEP_DISTANCE) * paper_distance_normalization - idx) * \
                                     (paperDistanceOffset[idx + 1] - paperDistanceOffset[idx])
            
            # 计算欧拉角
            R, _ = cv2.Rodrigues(rvec)
            paperPitch = math.atan2(R[2, 1], R[2, 2]) * 180 / math.pi
            paperYaw = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]** 2)) * 180 / math.pi
            paperRoll = math.atan2(R[1, 0], R[0, 0]) * 180 / math.pi


def euler2RVEC():
    """
    将欧拉角(pitch, yaw, roll)转换为旋转矩阵
    旋转顺序：ZYX (先yaw，再pitch，最后roll)
    :param pitch: X轴旋转角度(度)
    :param yaw: Z轴旋转角度(度)
    :param roll: Y轴旋转角度(度)
    :return: 3x3旋转矩阵
    """
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx
    # 转换为弧度
    pitch = math.radians(0.00)
    yaw = math.radians(0.00)
    roll = math.radians(0.00)
    
    # 计算各轴旋转矩阵
    # 绕Z轴旋转(yaw)
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 绕Y轴旋转(pitch)
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # 绕X轴旋转(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # 组合旋转矩阵 (ZYX顺序)
    R = Rz @ Ry @ Rx
    return R


def from3Dto2D(points_3d, camera_matrix, rvec, tvec, distortion=None):
    """
    将三维点投影到二维图像平面
    
    参数:
        points_3d: numpy数组，形状为(N,3)的3D点坐标
        camera_matrix: 相机内参矩阵，3x3 numpy数组
        rvec: 旋转向量，可以使用3x1 numpy数组
        tvec: 平移向量，3x1 numpy数组
        distortion: 畸变系数，可以为None或5x1 numpy数组
        
    返回:
        points_2d: 投影后的2D点坐标，形状为(N,2)的numpy数组
    """
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx
    if distortion is None:
        distortion = np.zeros((5, 1))
    
    # 如果rvec是旋转矩阵(3x3)，需要转换为旋转向量
    if rvec.shape == (3, 3):
        rvec, _ = cv2.Rodrigues(rvec)
    
    # 投影点
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, distortion)
    
    # 将结果从(N,1,2)转换为(N,2)
    return points_2d  # .reshape(-1, 2)


def PerspectiveTrans(img, pts1, pts2):
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    if len(pts1) == 4 and len(pts2) == 4:
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        ImgColorTrans = cv2.warpPerspective(img, matrix, (IMG_WIDTH, IMG_HEIGHT))


def drawCoordinateAxes(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                      rvec: np.ndarray, tvec: np.ndarray, length: float) -> None:
    """绘制三维坐标轴"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    axes_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    projected_points, _ = cv2.projectPoints(axes_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2).astype(np.int32)
    
    if len(quadsOrderPoints) == 4 and len(projected_points) == 4:
        cv2.line(image, tuple(projected_points[0]), tuple(projected_points[1]), (0, 0, 255), 4)  # X轴
        cv2.line(image, tuple(projected_points[0]), tuple(projected_points[2]), (0, 255, 0), 4)  # Y轴
        cv2.line(image, tuple(projected_points[0]), tuple(projected_points[3]), (255, 0, 0), 4)  # Z轴


def innerShapeGet(contours: List[np.ndarray], is_overlap: bool):
    """
    内部形状获取（0：三角形；1：正方形；2：圆形）

    参数:
    contours: 轮廓列表
    is_overlap: 是否存在重叠
    """
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    # 初始化
    innerShape = []
    innerShapeEdgeLength = []
    innerContours = []
    innerHierarchy = None
    innerRect = []
    innerPoints = []
    aiDetectionNum = 10
    aiDetectionROI = []
    aiDetectionNum2Idx = []
    innerSplitContours = []

    if len(contours) > 0 and ImgColorTrans is not None:
        # 设置ROI掩膜
        mask = np.zeros(ImgColor.shape[:2], np.uint8)
        cv2.drawContours(mask, contours, 0, 255, -1)

        # 复制图像到掩膜
        img_roi_gray = GE_Image.colorConvert(ImgColorTrans, cv2.COLOR_BGR2GRAY)
        img_roi_binary = GE_Image.BGR.thresholdConvert(img_roi_gray)
        img_roi_binary = cv2.bitwise_not(img_roi_binary)

        # 边缘提取
        # img_roi_edge = cv2.Canny(img_roi_gray, 100, 150, 3)

        # 释放并重新赋值ROI
        ImgROI = []
        ImgROI = cv2.bitwise_and(img_roi_binary, img_roi_binary, mask=mask)

        # 将轮廓点上的噪点遮盖
        for i in range(len(contours[0])):
            cv2.circle(ImgROI, tuple(contours[0][i][0]), 2, 0, -1)

        # 查找内部轮廓
        innerContours, innerHierarchy = GE_Image.findContours(
            ImgROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 找内部轮廓最小外接矩形
        innerRect = GE_Image.findContoursRect(innerContours)

        # 绘制内部轮廓
        if ImgTest is not None:
            # 绘制内部轮廓
            GE_Image.drawContours(ImgNano, innerContours, innerHierarchy, (255, 255, 0))
            GE_Image.drawContours(ImgTest, innerContours, innerHierarchy, (255, 255, 0))
            # 绘制内部轮廓最小外接矩形
            GE_Image.drawContoursRect(ImgNano, innerRect, (0, 255, 255))
            GE_Image.drawContoursRect(ImgTest, innerRect, (0, 255, 255))

        # 遍历内部轮廓
        for i in range(len(innerContours)):
            if i < len(innerContours) and i < len(innerRect):
                if innerContours[i] is not None and innerContours[i].size > 0:
                    ContourRectSquare = innerRect[i][1][0] * innerRect[i][1][1]
                    # 大于最小图形轮廓面积才能判断形状/在外侧/无重叠情况
                    if cv2.contourArea(innerContours[i]) > INNER_SHAPE_SQUARE_MIN and ContourRectSquare > INNER_SHAPE_SQUARE_MIN and not is_overlap:
                        innerSplitContours.append(innerContours[i])
                        # 计算轮廓周长
                        perimeter = cv2.arcLength(innerContours[i], True)
                        # 多边形逼近（epsilon为周长的一定比例）
                        inner_points_buf = cv2.approxPolyDP(
                            innerContours[i], 0.04 * perimeter, True)
                        innerPoints.append(inner_points_buf)
                        
                        # 调试点（对应原C++的debugPoint(71)）
                        # print("Debug point 71")
                        
                        # 找最小外接圆
                        (center_x, center_y), radius = cv2.minEnclosingCircle(innerContours[i])
                        # 计算轮廓面积和最小外接圆面积比值
                        square_contour = cv2.contourArea(innerContours[i])
                        square_rate = square_contour / (np.pi * radius * radius)
                        
                        # 形状判断
                        # 三角形
                        if len(inner_points_buf) == 3:
                            if square_rate < 0.85:  # 不是圆形
                                innerShape.append(0)
                                # 边长计算
                                innerShapeEdgeLength.append(perimeter / 3.0)
                            else:  # 是圆形
                                innerShape.append(2)
                                innerShapeEdgeLength.append(perimeter / np.pi)
                        
                        # 正方形
                        elif len(inner_points_buf) == 4:
                            if square_rate < 0.85:  # 不是圆形
                                # 计算轮廓面积和最小外接矩形面积比值
                                square_contour = cv2.contourArea(innerContours[i])
                                rect_area = innerRect[i][1][0] * innerRect[i][1][1]
                                square_rate = square_contour / rect_area
                                
                                # 比值大于阈值，判定为矩形
                                if square_rate > 0.85:
                                    innerShape.append(1)
                                    # 边长计算
                                    innerShapeEdgeLength.append(perimeter / 4.0)
                                else:
                                    isOverlap = True
                                    innerShapeGet(filterContoursNMS, isOverlap)
                            else:  # 是圆形
                                innerShape.append(2)
                                innerShapeEdgeLength.append(perimeter / np.pi)
                        
                        # 圆形（边数大于4）
                        elif len(inner_points_buf) > 4:
                            # 比值大于阈值，判定为圆形
                            if square_rate > 0.85:
                                innerShape.append(2)
                                innerShapeEdgeLength.append(perimeter / np.pi)
                            else:
                                isOverlap = True
                                innerShapeGet(filterContoursNMS, isOverlap)

                        box = cv2.boxPoints(innerRect[i])
                        from2_point_draw_rotated_square(ImgNano, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), (0, 0, 255))

                    # 处理重叠情况
                    elif cv2.contourArea(innerContours[i]) > INNER_SHAPE_SQUARE_MIN and ContourRectSquare > INNER_SHAPE_SQUARE_MIN and is_overlap:
                        # 计算轮廓周长
                        perimeter = cv2.arcLength(innerContours[i], True)
                        # 多边形逼近
                        inner_points_buf = cv2.approxPolyDP(
                            innerContours[i], 0.008 * perimeter, True)
                        innerPoints.append(inner_points_buf)
                        
                        # 调试点（对应原C++的debugPoint(72)）
                        # print("Debug point 72")
                        
                        inner_point_kind = []
                        # 点分类，分为边框角点和相交点
                        for j in range(len(inner_points_buf)):
                            point = (int(inner_points_buf[j][0][0]), int(inner_points_buf[j][0][1]))
                            # 创建圆形掩模
                            mask = np.zeros(ImgGray.shape[:2], np.uint8)
                            cv2.circle(mask, point, POINT_SORT_CIRCLE_ROI_RADIUS, 255, -1)
                            
                            # 提取圆形ROI
                            ImgGrayTrans = GE_Image.colorConvert(ImgColorTrans, cv2.COLOR_BGR2GRAY)
                            img_circle_roi_gray = cv2.bitwise_and(ImgGrayTrans, ImgGrayTrans, mask=mask)
                            
                            # 二值化
                            img_circle_roi_binary = GE_Image.BGR.thresholdConvert(img_circle_roi_gray)
                            
                            # 像素统计
                            white_pixels = cv2.countNonZero(img_circle_roi_binary)
                            total_pixels = cv2.countNonZero(mask)
                            
                            # 计算黑色像素比例
                            if total_pixels > 0:
                                black_ratio = 1.0 - (white_pixels / total_pixels)
                                black_ratio = round(black_ratio * 100) / 100  # 保留两位小数
                            else:
                                black_ratio = 0.0
                                
                            # 点类型分类
                            inner_point_kind.append(black_ratio <= POINT_SORT_LIMIT)
                        
                        overlapRectExtract(inner_points_buf, inner_point_kind)
                        innerPointsKind.append(inner_point_kind)  # 存储点类型
    mask = np.zeros(ImgColor.shape[:2], np.uint8)


def overlapRectExtract(inner_point: List[Tuple[float, float]], inner_point_kind: List[bool]) -> None:
    """重叠矩形提取"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx
    
    innerShape = []
    innerContours = []
    innerHierarchy = []
    innerSplitContours = []
    innerSplitHierarchy = []
    innerSplitPoints = []
    innerSplitTruePoints = []
    innerSplitPointsKind = []
    innerShapeEdgeLength = []

    if len(inner_point) == len(inner_point_kind):
        # 使相交点排第一个
        for i in range(len(inner_point)):
            if not inner_point_kind[i]:
                inner_point = np.roll(inner_point, -i, axis=0)
                inner_point_kind = np.roll(inner_point_kind, -i, axis=0)
                break
        # 内部图形矩形交点
        inner_rect_intersection_point = []
        true_point_num = 0
        # 判断矩形上交点是否属于同一个矩形
        for i in range(len(inner_point) + 1):
            idx = i % len(inner_point)
            if not inner_point_kind[idx]:
                inner_rect_intersection_point.append(inner_point[idx])
                if true_point_num == 3:
                    cv2.line(ImgROI, 
                            inner_rect_intersection_point[len(inner_rect_intersection_point)-1][0], 
                            inner_rect_intersection_point[len(inner_rect_intersection_point)-2][0], 
                            (0), 4)
                    cv2.line(ImgNano, 
                            inner_rect_intersection_point[len(inner_rect_intersection_point)-1][0], 
                            inner_rect_intersection_point[len(inner_rect_intersection_point)-2][0], 
                            (255,255,255), 3)
                true_point_num = 0
            else:
                true_point_num += 1
        
        # 查找所有分割轮廓
        innerSplitContours, innerSplitHierarchy = GE_Image.findContours(ImgROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 调试
        # print(len(innerSplitContours))
        # print("AREA")
        # for i in range(len(innerSplitContours)):
        #     if i < len(innerSplitContours):
        #         print(cv2.contourArea(innerSplitContours[i]))
        # 遍历所有分割图形轮廓
        for i in range(len(innerSplitContours)):
            if i < len(innerSplitContours):
                contour = innerSplitContours[i]
                if cv2.contourArea(contour) > INNER_SHAPE_SQUARE_MIN:
                    # 再次在内部轮廓内进行取ROI，防止隔壁的内部图形对点分类造成影响
                    # 设置ROI掩膜
                    img_roi_binary = np.zeros(ImgColor.shape[:2], np.uint8)
                    cv2.drawContours(img_roi_binary, [contour], 0, 255, -1)
                    
                    # 多边形近似
                    perimeter = cv2.arcLength(contour, True)
                    # 当内部轮廓近似多边形缺失角点时改小，多角点时改大
                    inner_split_points_buf = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
                    innerSplitPoints.append(inner_split_points_buf)
                    
                    inner_split_point_kind = []
                    inner_split_true_points_buf = []
                    true_split_point_num = 0
                    # 点分类，分为边框角点和相交点
                    for p in inner_split_points_buf:
                        point = (p[0][0], p[0][1])
                        # 创建圆形掩膜
                        mask = np.zeros(ImgGray.shape[:2], dtype=np.uint8)
                        cv2.circle(mask, point, 20, 255, -1)

                        # 提取圆形ROI
                        img_circle_roi_binary = cv2.bitwise_and(img_roi_binary, img_roi_binary, mask=mask)

                        # 像素统计
                        white_pixels = cv2.countNonZero(img_circle_roi_binary)
                        total_pixels = cv2.countNonZero(mask)
                        white_ratio = white_pixels / total_pixels if total_pixels != 0 else 0.0
                        
                        # 需要调：当分割矩形识别不出来时，修改 black_ratio
                        if white_ratio <= POINT_SORT_LIMIT:
                            inner_split_point_kind.append(True)
                            inner_split_true_points_buf.append(point)
                            true_split_point_num += 1
                            cv2.circle(ImgTest, p[0], 6, (0, 255, 0), -1)
                        else:
                            inner_split_point_kind.append(False)
                            cv2.circle(ImgTest, p[0], 6, (0, 0, 255), -1)
                    
                    # print(true_split_point_num)
                    
                    # 处理分割图形
                    if true_split_point_num == 2:
                        innerShape.append(1)
                        edge_length = cv2.norm(np.array(inner_split_true_points_buf[0]) - np.array(inner_split_true_points_buf[1])) / 1.4142135623730950488016887242
                        innerShapeEdgeLength.append(edge_length)
                        # 绘制矩形边框角点
                        # for pt in inner_split_true_points_buf:
                        #     cv2.circle(ImgNano, pt, 6, (0, 0, 255), -1)
                        # 绘制矩形边框
                        from2_point_draw_rotated_square(ImgNano, inner_split_true_points_buf[0], inner_split_true_points_buf[1], (0, 255, 0), (0, 255, 0))
                    # 绘制正方形（简化实现）
                    elif true_split_point_num == 3:
                        max_dist = 0
                        p1 = None
                        p2 = None
                        # 计算对角线点
                        for i in range(3):
                            dist = cv2.norm(np.array(inner_split_true_points_buf[i]) - np.array(inner_split_true_points_buf[(i+1)%3]))
                            # 找对角线
                            if dist > max_dist:
                                max_dist = dist
                                p1 = inner_split_true_points_buf[i]
                                p2 = inner_split_true_points_buf[(i+1)%3]
                        innerShape.append(1)
                        edge_length = max_dist / 1.4142135623730950488016887242
                        innerShapeEdgeLength.append(edge_length)
                        # 绘制矩形边框角点
                        # for pt in inner_split_true_points_buf:
                            # cv2.circle(ImgNano, pt, 6, (0, 0, 255), -1)
                        # 绘制矩形边框
                        from2_point_draw_rotated_square(ImgNano, p1, p2, (0, 255, 0), (0, 255, 0))
                        
                    
                    innerSplitPointsKind.append(inner_split_point_kind)
                    innerSplitTruePoints.append(inner_split_true_points_buf)
            
        GE_Image.drawContours(ImgTest, innerSplitContours, innerSplitHierarchy, (255, 255, 0))


def from2_point_draw_rotated_square(image, pt1, pt2, colorRect, colorText):
    """
    根据    根据对角线两点绘制旋转正方形
    
    参数:
        image: 要绘制的图像
        pt1: 对角线第一个点 (x, y)
        pt2: 对角线第二个点 (x, y)
        color: 绘制颜色，格式为 (B, G, R)
    """
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    # 计算中心点
    center = ((pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0)
    
    # 计算对角线长度
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    diagonal = math.hypot(dx, dy)
    side = diagonal / math.sqrt(2)  # 计算正方形边长
    
    # 计算旋转角度（以pt1到pt2的方向为基准）
    angle = math.atan2(dy, dx) * 180 / math.pi
    
    # 创建旋转矩形
    rotated_rect = ((center[0], center[1]), (side, side), angle - 45)
    
    # 获取四个顶点
    vertices = cv2.boxPoints(rotated_rect)
    vertices = np.intp(vertices)  # 转换为整数坐标
    
    # 添加AI识别ROI
    x, y, w, h = cv2.boundingRect(vertices)
    aiDetectionROI.append((x, y, w, h))
    
    text_x = 0
    text_y = 0

    # 绘制正方形
    for i in range(4):
        cv2.line(image, tuple(vertices[i]), tuple(vertices[(i + 1) % 4]), colorRect, 2)

    # 在中心点绘制文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    # 获取文字大小以居中显示
    text = str(len(innerShape)-1)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    center_x = (pt1[0] + pt2[0]) // 2
    center_y = (pt1[1] + pt2[1]) // 2
    # 调整文字位置使其真正居中
    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, colorText, thickness)


def AI(contours: List[np.ndarray]) -> None:
    """AI识别"""
    global ImgColor, ImgColorTrans, ImgROI, ImgNano, ImgGray, ImgGrayTrans, ImgTest, ImgBinary, ImgBinaryTrans, ImgEdge
    global allContours, allRect, allCenter, allHierarchy, filterContours, filterRect, filterCenter
    global filterContoursNMS, filterRectNMS, filterCenterNMS, quadsPoints, quadsOrderPoints, quadsEdgeLength
    global quadsEdgePerimeter, innerContours, innerHierarchy, innerPoints, innerPointsKind, innerSplitPoints, innerSplitTruePoints, innerSplitPointsKind, paperDistanceRaw, paperDistanceCalibration
    global paperYaw, paperPitch, paperRoll, isOverlap, rvec, tvec, innerShape, innerShapeEdgeLength
    global innerSplitContours, innerSplitHierarchy, neededNum, aiDetectionNum, aiDetectionROI, aiDetectionNum2Idx

    for i in range(len(aiDetectionROI)):
        if i < len(aiDetectionROI) and i < len(innerSplitContours):
            result = 10  # AI识别结果

            #------------识别代码------------#
            
            CENTER_RATIO = 0.9  # 中心区域占ROI的比例（60%）
            x,y,w,h = aiDetectionROI[i]


            # 设置ROI掩膜
            mask = np.zeros(ImgColor.shape[:2], np.uint8)
            cv2.drawContours(mask, contours, 0, 255, -1)
            # 复制图像到掩膜
            ImgGrayAI = GE_Image.colorConvert(ImgColorTrans, cv2.COLOR_BGR2GRAY)
            # ImgGrayAI = GE_Image.Dilate(ImgGrayAI, 3)  # 膨胀，黑色变少
            # 释放并重新赋值ROI
            ImgAiROI = []
            ImgAiROI = cv2.bitwise_and(ImgGrayAI, ImgGrayAI, mask=mask)
            ImgAiROI = GE_Image.BGR.thresholdConvert(ImgAiROI)

            # disp.show(image.cv2image(ImgAiROI))
            
            # 构造ROI轮廓
            # 设置ROI掩膜
            mask = np.zeros(ImgAiROI.shape[:2], np.uint8)
            cv2.drawContours(mask, innerSplitContours, i, 255, -1)

            ImgAiROI = cv2.bitwise_and(ImgAiROI, ImgAiROI, mask=mask)
            # 将轮廓点上的噪点遮盖
            for j in range(len(innerSplitContours[i])):
                cv2.circle(ImgAiROI, tuple(innerSplitContours[i][j][0]), 3, 0, -1)

            # 放大区域
            ImgAiROIBig = ImgAiROI[y:y+h, x:x+w]
            ImgAiROI = cv2.resize(ImgAiROIBig, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            # 获取图像尺寸
            height, width = ImgAiROI.shape[:2]

            # 调试
            # ImgBlack = np.zeros((width, height), np.uint8)
            # ImgBlack[0:width, 0:height] = ImgAiROI
            # ImgBlack = cv2.resize(ImgBlack, (IMG_WIDTH, IMG_HEIGHT))
            # disp.show(image.cv2image(ImgBlack))

            ImgAiROI_maix = image.cv2image(ImgAiROI)
            img_crop = ImgAiROI_maix.crop(0,0,width,height)
            img_roi_final = img_crop.to_format(image.Format.FMT_RGB888)

            #调试用的
            # cv2.line(ImgBlack, (width, 0), (width, height), (255), 1)
            # cv2.line(ImgBlack, (0, height), (width, height), (255), 1)
            # disp.show(image.cv2image(ImgBlack))

            # img_roi_cv_resize = cv2.resize(img_roi_cv,(64,64))
            # disp.show(image.cv2image(img_detect))
            

            # img_roi_detect = image.cv2image(img_roi_cv_resize)

            # img_roi_scaled = img_roi.resize(w*2, h*2)
            # img_roi_rgb = img_roi_scaled.to_format(image.Format.FMT_RGB888)

            objs = detector.detect(img_roi_final,conf_th = 0.5,iou_th = 0.45)

            for obj in objs:

                # 计算检测框的中心坐标（相对于原图）
                #还要写加一个判断范围，在正中间才识别，或者才输出
                obj_center_x = x + obj.x + obj.w // 2
                obj_center_y = y + obj.y + obj.h // 2

                # if (center_x_min <= obj_center_x <= center_x_max and 
                #         center_y_min <= obj_center_y <= center_y_max):

                cv2.rectangle(ImgColor, (int(x + obj.x), int(y + obj.y)), (int(x + obj.x + obj.w), int(y + obj.y + obj.h)),(0, 0, 255),2)  # 线宽
                msg = f'{detector.labels[obj.class_id]}:{obj.score:.2f}'
                result = obj.class_id   #存储识别结果
                cv2.putText(ImgColor,  msg, (int(x + obj.x), int(y + obj.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255),  1,cv2.LINE_AA)  # 抗锯齿
                
            #------------识别代码------------#

            aiDetectionNum = result
            aiDetectionNum2Idx.append((aiDetectionNum, i))


def debugPoint(step: int) -> None:
    """DEBUG软件打点"""
    if DEBUG_EN:
        print(f"STEP: {step}")

def uartGetChar():
    uart_num = serial0.read()  # 获取当前串口数据数量
    if uart_num:
        uart_str = uart_num.decode()
    else:
        uart_str = 0
    return uart_str


# 主函数相关常量
WIDTH_640 = 640
HEIGHT_480 = 480
V4L2 = 0
FPS_60 = 60
EXP_6 = 6
MJPG = 0

# 打开屏幕显示
disp = display.Display()
# 设置摄像头
cam = camera.Camera(width=640, height=480)
# 帧数计时器
last_time = time.time()
# 图像处理
GE_Image = Image()
# 串口
#----------------------------------------------------------------------------#
# pinmap.set_pin_function("A18", "UART1_RX")
# pinmap.set_pin_function("A19", "UART1_TX")


# device = "/dev/ttyS0"
# # ports = uart.list_devices() # 列出当前可用的串口
# serial0 = uart.UART(device, 115200)


pinmap.set_pin_function("A29", "UART2_RX")
pinmap.set_pin_function("A28", "UART2_TX")
device2 = "/dev/ttyS2"
serial1 = uart.UART(device2, 115200)

pinmap.set_pin_function("A18", "UART1_RX")
pinmap.set_pin_function("A19", "UART1_TX")
device0 = "/dev/ttyS1"
serial0 = uart.UART(device0, 115200)

#----------------------------------------------------------------------------#
def main():
    frame_num = 0
    
    while True:
        imgProcess()
        initVariable()
        # 串口
        uartDataNum = 0
        uartData = serial0.read()  # 获取当前串口数据数量
        if GAME_EN == True: # 比赛使能：读取串口才执行程序
            if len(uartData) != 0:
                uartDataNum = len(uartData)
            else:
                uartDataNum = 0
        else:   # 比赛失能
            uartDataNum = 1
            neededNum = 2

        # 识别
        if uartDataNum != 0:
            if uartDataNum == 1 and GAME_EN == True:
                if isinstance(uartData[0], int):  # bytes/bytearray 的 uartData[0] 是 int
                    neededNum = uartData[0] # int(format(ord(uartData[0]), '02x'), 16)
                    # print(neededNum)
            start_time = time.time()
            
            debugPoint(1)
            # 查找轮廓
            allContours, allHierarchy = GE_Image.findContours(ImgBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # print(len(allContours))

            debugPoint(2)
            # 筛选轮廓
            contourFilter(allContours)
            GE_Image.drawContours(ImgNano, filterContours, allHierarchy, (255, 0, 0))
            # print(len(filterContours))
            
            debugPoint(3)
            # 轮廓非极大值抑制
            contourNMS(filterContours)
            GE_Image.drawContours(ImgNano, filterContoursNMS, allHierarchy, (0, 255, 0))
            # print(len(filterContoursNMS))
            
            debugPoint(4)
            # 绘制筛选轮廓
            GE_Image.drawContours(ImgNano, filterContoursNMS, allHierarchy, (0, 0, 255))
            
            debugPoint(5)
            # 将轮廓转为四边形四个点
            Contours2Quads(filterContoursNMS, filterCenterNMS)
            
            debugPoint(8)

            # 计算距离与欧拉角
            calu4DistanceAndEularAngle(cameraMatrix, distCoeffs)

            debugPoint(9)

            # 计算正视时的旋转矩阵
            revcTarget = euler2RVEC()

            debugPoint(10)

            # 计算正视时的二维坐标
            imgObjectPointsTarget = from3Dto2D(objectPoints, cameraMatrix, revcTarget, tvec, distCoeffs)

            debugPoint(11)

            # 透视变换
            PerspectiveTrans(ImgColor, quadsOrderPoints, imgObjectPointsTarget)

            debugPoint(12)

            ImgGrayTrans = GE_Image.colorConvert(ImgColorTrans, cv2.COLOR_BGR2GRAY)

            # 图像腐蚀膨胀
            ImgGrayTransClosing = GE_Image.Dilate(ImgGrayTrans, DILATE_CORE_SIZE)   # 黑色变少
            ImgGrayTransClosing = GE_Image.Erode(ImgGrayTransClosing, ERODE_CORE_SIZE)   # 黑色变多
            ImgBinaryTrans = GE_Image.BGR.thresholdConvert(ImgGrayTransClosing)
            # 查找轮廓
            allContours, allHierarchy = GE_Image.findContours(ImgBinaryTrans, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # print(len(allContours))

            debugPoint(13)
            # 筛选轮廓
            contourFilter(allContours)
            GE_Image.drawContours(ImgNano, filterContours, allHierarchy, (255, 0, 0))
            # print(len(filterContours))
            
            debugPoint(14)
            # 轮廓非极大值抑制
            contourNMS(filterContours)
            GE_Image.drawContours(ImgNano, filterContoursNMS, allHierarchy, (0, 255, 0))
            # print(len(filterContoursNMS))
            
            debugPoint(15)
            # 绘制筛选轮廓
            GE_Image.drawContours(ImgNano, filterContoursNMS, allHierarchy, (0, 0, 255))
            
            debugPoint(16)
            # 将轮廓转为四边形四个点
            Contours2Quads(filterContoursNMS, filterCenterNMS)
            
            debugPoint(17)
            # 绘制轮廓、角点、序号、边长
            for i in range(len(quadsOrderPoints)):
                if i < len(quadsOrderPoints) and i < len(quadsEdgeLength): 
                    cv2.circle(ImgNano, quadsOrderPoints[i], 10, (0, 0, 255), 3)
                    distance_str = str(quadsEdgeLength[i])
                    x = (quadsOrderPoints[(i+1)%4][0] + quadsOrderPoints[i][0]) // 2
                    y = (quadsOrderPoints[(i+1)%4][1] + quadsOrderPoints[i][1]) // 2
                    cv2.putText(ImgNano, distance_str, (x, y), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
                
                    text = str(i + 1)
                    cv2.putText(ImgNano, text, quadsOrderPoints[i], cv2.FONT_ITALIC, 0.5, (0, 255, 0))
                    cv2.line(ImgNano, quadsOrderPoints[i], quadsOrderPoints[(i+1)%4], (0, 255, 0), 1)

            # 获取内部图形形状
            innerShapeGet(filterContoursNMS, isOverlap)

            debugPoint(18)

            # AI识别
            if neededNum != 10:
                AI(filterContoursNMS)
            
            debugPoint(19)

            # 绘制内部图形点
            for i in range(len(innerPoints)):
                # 遍历每个内部轮廓的所有点
                for j in range(len(innerPoints[i])):
                    # 获取当前点和下一个点（最后一个点连接回第一个点）
                    point1 = (int(innerPoints[i][j][0][0]), int(innerPoints[i][j][0][1]))
                    next_j = (j + 1) % len(innerPoints[i])
                    point2 = (int(innerPoints[i][next_j][0][0]), int(innerPoints[i][next_j][0][1]))
                    
                    # 如果存在重叠，根据点的类型绘制不同颜色的圆
                    if isOverlap:
                        if innerPointsKind[i][j]:
                            # 绿色圆表示一种类型的点
                            cv2.circle(ImgNano, point1, 6, (0, 255, 0), 1)
                        else:
                            # 红色圆表示另一种类型的点
                            cv2.circle(ImgNano, point1, 6, (0, 0, 255), 1)
                    else:
                        # 绘制连接点的线
                        cv2.line(ImgNano, innerPoints[i][j][0], innerPoints[i][(j+1)%len(innerPoints[i])][0], (0, 255, 0), 1)
                        # 红色圆表示角点
                        cv2.circle(ImgNano, point1, 10, (0, 0, 255), 3)

            debugPoint(20)

            # 绘制三维坐标轴
            # drawCoordinateAxes(ImgNano, cameraMatrix, distCoeffs, rvec, tvec, 200.0)
            
            debugPoint(21)
            
            frame_num += 1
            
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            
            if(PRINT_EN):       # 终端打印
                print("**************************************************************")
                print(f"FRAME MUN: {frame_num}    FPS: {fps:.2f}")
                print(f"DISTANCE RAW: {paperDistanceRaw:.2f}")
                print(f"DISTANCE CALIBRATION: {paperDistanceCalibration:.2f}")
                print(f"OVERLAP STATUS: {'TRUE' if isOverlap else 'false'}")
                print(f"NUM2IDX: {aiDetectionNum2Idx}")
                if len(quadsEdgeLength) == 4:
                    # 遍历所有图形
                    for i in range(len(innerShape)):
                        if i < len(innerShape):
                            if innerShape[i] == 0:
                                print("SHAPE: TRIANGLE")
                            elif innerShape[i] == 1:
                                print("SHAPE: SQUARE")
                            elif innerShape[i] == 2:
                                print("SHAPE: CIRCLE")

                            edge_length = 2.0 * (CALIBRATION_HEIGHT + CALIBRATION_WIDTH) * innerShapeEdgeLength[i] / quadsEdgePerimeter
                            print(f"EDGE LENGTH: {edge_length:.2f}")
                            
                            if innerShape[i] in [0, 1, 2] and i < len(innerShapeEdgeLength):
                                if innerShape[i] == 0:
                                    innerShapeSquare = edge_length * edge_length * math.sqrt(3) / 4
                                    print(f"SQUARE: {innerShapeSquare:.2f}")
                                elif innerShape[i] == 1:
                                    innerShapeSquare = edge_length * edge_length
                                    print(f"SQUARE: {innerShapeSquare:.2f}")
                                else:
                                    innerShapeSquare = math.pi * edge_length * edge_length
                                    print(f"SQUARE: {innerShapeSquare:.2f}")
                    print(f"YAW: {paperYaw:.2f}")
                    print(f"PITCH: {paperPitch:.2f}")
                    print(f"ROLL: {paperRoll:.2f}")
                print("**************************************************************")

            if(UART_EN):        # 串口发送
                if len(quadsEdgeLength) == 4:
                    # 计算最小面积
                    minInnerShapeSquare = IMG_WIDTH*IMG_HEIGHT  # 最小内部图形面积
                    innerShapeSquare = 0    # 内部图形面积
                    idx = 0 # 图形索引
                    minInnerShapeSquare = IMG_WIDTH*IMG_HEIGHT  # 最小内部图形面积
                    innerShapeSquare = 0    # 内部图形面积

                    # 计算所有图形的边长
                    edgeLength = []
                    uartData = ''
                    for i in range(len(innerShape)):
                        # 计算边长
                        edgeLength.append(2.0 * (CALIBRATION_HEIGHT + CALIBRATION_WIDTH) * innerShapeEdgeLength[i] / quadsEdgePerimeter)

                    if neededNum == 10:
                        # 遍历所有图形，找最小面积的图形
                        for i in range(len(innerShape)):
                            if i < len(innerShape):
                                # 计算最小面积图形的索引
                                if innerShape[i] in [0, 1, 2] and i < len(innerShapeEdgeLength):
                                    if innerShape[i] == 0:
                                        innerShapeSquare = edgeLength[i] * edgeLength[i] * math.sqrt(3) / 4
                                        if(innerShapeSquare < minInnerShapeSquare):
                                            minInnerShapeSquare = innerShapeSquare
                                            idx = i
                                    elif innerShape[i] == 1:
                                        innerShapeSquare = edgeLength[i] * edgeLength[i]
                                        if(innerShapeSquare < minInnerShapeSquare):
                                            minInnerShapeSquare = innerShapeSquare
                                            idx = i
                                    else:
                                        innerShapeSquare = math.pi * edgeLength[i] * edgeLength[i]
                                        if(innerShapeSquare < minInnerShapeSquare):
                                            minInnerShapeSquare = innerShapeSquare
                                            idx = i
                        if idx < len(innerShape):
                            uartData = f"[{round(paperDistanceCalibration)},{round(innerShape[idx])}],[{round(edgeLength[idx])},{round(minInnerShapeSquare)}]\n"  # 添加换行符作为分隔符
                    elif 0 <= neededNum < 10:
                        # 遍历AI结果和框索引的映射
                        for i in range(len(aiDetectionNum2Idx)):
                            if aiDetectionNum2Idx[i][0] == neededNum:
                                idx = aiDetectionNum2Idx[i][1]
                                break
                        # 防止idx越界
                        if idx < len(aiDetectionNum2Idx) and idx < len(edgeLength) and idx < len(innerShape):
                            # 计算指定索引图形的面积
                            if innerShape[idx] == 0:
                                innerShapeSquare = edgeLength[idx] * edgeLength[idx] * math.sqrt(3) / 4
                            elif innerShape[idx] == 1:
                                innerShapeSquare = edgeLength[idx] * edgeLength[idx]
                            else:
                                innerShapeSquare = math.pi * edgeLength[idx] * edgeLength[idx]
                            uartData = f"[{round(paperDistanceCalibration)},{round(innerShape[idx])}],[{round(edgeLength[idx])},{round(innerShapeSquare)}]\n"  # 添加换行符作为分隔符
                    serial0.write(uartData.encode('utf-8'))  # 需要编码为bytes

                    # 遍历AI结果和框索引的映射（调试用）
                    # print(aiDetectionNum2Idx)
                    # print(neededNum)
                    # for i in range(len(aiDetectionNum2Idx)):
                    #     if aiDetectionNum2Idx[i][0] == neededNum:
                    #         idx = aiDetectionNum2Idx[i][1]
                    #         print(idx)
                    #         break

        disp.show(image.cv2image(np.array(ImgNano, dtype=np.uint8)))
        # disp.show(image.cv2image(ImgEdge))


if __name__ == "__main__":
    main()