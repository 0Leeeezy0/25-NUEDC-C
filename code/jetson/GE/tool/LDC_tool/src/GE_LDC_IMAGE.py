import cv2
import numpy as np
import os

# 全局变量
board_size = (8, 6)        # 棋盘格内角点数量 (宽,高)
square_size = 24.0         # 每个方格的实际尺寸(mm)
image_points = []          # 存储所有图像的角点
object_points = []         # 存储所有3D点
camera_matrix = None
dist_coeffs = None
image_size = None
images_count = 0
min_images = 23            # 最少需要采集的图像数
img_file_path = "../img/STD/2025-07-31/"  # 已经采集的图像路径
img_file_format = ".jpg"   # 已经采集的图像文件格式

def init_object_points():
    """初始化3D对象点"""
    obj = []
    for i in range(board_size[1]):
        for j in range(board_size[0]):
            obj.append([j * square_size, i * square_size, 0])
    return np.array(obj, dtype=np.float32)

def detect_chessboard_corners(image):
    """检测棋盘格角点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners
    return ret, None

def perform_camera_calibration():
    """执行相机标定"""
    global camera_matrix, dist_coeffs
    
    if len(image_points) < min_images:
        print(f"需要至少 {min_images} 张有效图像，当前只有 {len(image_points)} 张")
        return False
    
    # 初始化对象点（所有图像使用相同的3D点）
    obj_points = [init_object_points() for _ in range(len(image_points))]
    
    # 标定相机
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, image_points, image_size, None, None, flags=flags)
    
    print("\n标定结果:")
    print(f"RMS误差: {ret} 像素")
    print("相机内参矩阵:")
    print(camera_matrix)
    print("畸变系数:")
    print(dist_coeffs)
    
    # 保存标定结果
    np.savez("camera_calibration.npz", 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs, 
             rms_error=ret)
    print("标定结果已保存到 camera_calibration.npz")
    return True

def main():
    global image_size, images_count, min_images
    
    # 获取第一帧以确定图像尺寸
    first_frame_path = os.path.join(img_file_path, f"0{img_file_format}")
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print("无法读取初始图像!")
        return
    
    image_size = (first_frame.shape[1], first_frame.shape[0])
    
    print("相机标定程序")
    print("使用方法:")
    print(f"1. 准备 {board_size[0]-1}x{board_size[1]-1} 的棋盘格")
    
    cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
    
    num = 1
    while True:
        # 读取图像
        img_path = os.path.join(img_file_path, f"{num}{img_file_format}")
        frame = cv2.imread(img_path)
        if frame is None:
            break
        
        # 尝试检测棋盘格
        found, corners = detect_chessboard_corners(frame)
        
        # 显示结果
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, board_size, corners, found)
            cv2.putText(view, f"GET: {images_count}/{min_images}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(view, "NO FOUND", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Camera Calibration", view)
        
        if found:
            image_points.append(corners)
            images_count += 1
            print(f"已捕获图像 {images_count}")
            
            # 短暂显示绿色边框表示捕获成功
            cv2.rectangle(view, (0, 0), (view.shape[1]-1, view.shape[0]-1), (0, 255, 0), 5)
            cv2.imshow("Camera Calibration", view)
            cv2.waitKey(300)
        else:
            # 未找到角点时减少所需图像数量
            min_images = max(2, min_images - 1)
        
        num += 1
        # 按ESC键提前退出
        if cv2.waitKey(1) == 27:
            break
    
    # 执行标定
    if image_points:
        calibration_success = perform_camera_calibration()
        
        if calibration_success:
            # 显示标定后的效果
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None,
                cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 1),
                image_size, cv2.CV_16SC2)
            
            print("\n按任意键查看去畸变效果，按'q'退出...")
            while True:
                undistorted = cv2.remap(first_frame, map1, map2, cv2.INTER_LINEAR)
                cv2.imshow("原始图像", first_frame)
                cv2.imshow("去畸变图像", undistorted)
                
                key = cv2.waitKey(30)
                if key >= 0:
                    if chr(key & 0xFF) == 'q':
                        break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()