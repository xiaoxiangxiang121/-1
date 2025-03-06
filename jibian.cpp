#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

int main_2(int argc, char** argv)
{
    // 文件路径
    std::string left_file = "E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\Grab_211020.bmp";
    std::string right_file = "E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\Grab_211022.bmp";
    // 读取左右两张图像
    cv::Mat left_image = cv::imread(left_file, 0);
    cv::Mat right_image = cv::imread(right_file, 0);

    ///相机参数部分
    // 创建左右相机的相机矩阵
    cv::Mat camera_matrix1 = (cv::Mat_<double>(3, 3) <<
        3248.2158, 0.8975, 1915.5583,
        0.0,3270.8959, 2025.6159,
        0.0, 0.0, 1.0);

    cv::Mat camera_matrix2 = (cv::Mat_<double>(3, 3) <<
        3247.2809, -8.2370, 1964.6075,
        0.0, 3253.6488, 1951.0575,
        0.0, 0.0, 1.0);

    // 创建左右相机的畸变参数
    cv::Mat dist_coeffs1 = (cv::Mat_<double>(1, 4) <<
        -0.0964, -0.0726, 0.00593, 0.0005);

    cv::Mat dist_coeffs2 = (cv::Mat_<double>(1, 4) <<
        -0.2222, 0.3863, 0.0031, 0.00204);

    // 创建左右相机的旋转矩阵和平移矩阵
    cv::Mat R, T;
    R = (cv::Mat_<double>(3, 3) <<
        0.9998,-0.0002,0.0207,
        0.0003,1.0000,-0.0060,
        -0.0207,0.0060,0.9998);
    cv::transpose(R, R);

    T = (cv::Mat_<double>(3, 1) << -116.2386, -1.3485, -0.8361);
    ///

    ///去畸变
    cv::Mat undistorted_left, undistorted_right;
    cv::undistort(left_image, undistorted_left, camera_matrix1, dist_coeffs1);
    cv::undistort(right_image, undistorted_right, camera_matrix2, dist_coeffs2);

    //保存去畸变后的图像
    cv::imwrite("E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\undistorted_left.jpg", undistorted_left);
    cv::imwrite("E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\undistorted_right.jpg", undistorted_right);
    ///

    ///立体校正
    // 根据标定结果创建左右摄像头的投影矩阵
    cv::Mat R1, P1, R2, P2, Q;
    // 进行立体校正
    cv::stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, left_image.size(), R, T, R1, R2, P1, P2, Q);

    // 创建校正映射
    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, left_image.size(), CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, left_image.size(), CV_32FC1, map2x, map2y);

    // 应用校正映射
    cv::Mat rectified_left, rectified_right;
    cv::remap(left_image, rectified_left, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(right_image, rectified_right, map2x, map2y, cv::INTER_LINEAR);

   
    // 保存校正后的图像
    cv::resize(rectified_left, rectified_left, cv::Size(), 0.2, 0.2);  // 缩小为原来的 50%
    cv::resize(rectified_right, rectified_right, cv::Size(), 0.2, 0.2);  // 缩小为原来的 50%

    // 画一些对齐的线
    //for (int y = 100; y < rectified_left.rows; y += 40) {
    //    cv::line(rectified_left, cv::Point(0, y), cv::Point(rectified_left.cols, y), cv::Scalar(255, 0, 0), 2);
    //    cv::line(rectified_right, cv::Point(0, y), cv::Point(rectified_right.cols, y), cv::Scalar(255, 0, 0), 2);
    //}
    cv::imshow("rectified_left", rectified_left);
    cv::imshow("rectified_right", rectified_right);
    cv::imwrite("E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\rectified_left.jpg", rectified_left);
    cv::imwrite("E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\rectified_right.jpg", rectified_right);
    ///

    /// 计算视差图并绘制对齐的线
    cv::Mat disparity;
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 9); // 使用Block Matching算法
    stereo->compute(rectified_left, rectified_right, disparity);

    // 归一化视差图显示
    cv::Mat disparity_normalized;
    disparity.convertTo(disparity_normalized, CV_8U, 255 / (16.0 * 16.0));
    cv::imshow("Disparity", disparity_normalized);

    

    cv::waitKey(0);
    return 0;
}
