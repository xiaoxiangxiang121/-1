#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h> 
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <string>

using namespace std;
using namespace pcl;
using namespace cv;
// 文件路径
std::string left_file = "E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\rectified_left.jpg";
std::string right_file = "E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\rectified_right.jpg";

int main()
{

    // 双目相机参数
    //double fx = 3031.8, fy = 3033.664, cx = 1900.0, cy = 2042.97;
    //double b = 140.0327;
    //double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    //double b = 0.573;
    double fx = 3248.2158*0.2, fy = 3270.8959*0.2, cx = 1915.5583*0.2, cy = 2025.6159*0.2;
    double b = 118.2494;//116.2494

   
    // 读取左右两目图像并计算视差
    cv::Mat left = cv::imread(left_file, cv::IMREAD_GRAYSCALE);
    if (left.empty()) {
        std::cerr << "Error: Could not open or find the left image: " << left_file << std::endl;
        return -1;
    }
    cv::Mat right = cv::imread(right_file, cv::IMREAD_GRAYSCALE);
    if (right.empty()) {
        std::cerr << "Error: Could not open or find the right image: " << right_file << std::endl;
        return -1;
    }

    

    



 

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);

    //将disparity_sgbm图像转换为32位浮点型（CV_32F）的图像，并将像素值从16位整型转换为浮点型。
    // 转换因子1.0 / 16.0f用于将16位整型的视差值缩放到0到1之间的浮点数，这样做可以提高视差图的精度，
    // 使其更易于处理和可视化。
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);



    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(left, cv::noArray(), keypoints, descriptors);
    cv::Mat outputImage;
    cv::drawKeypoints(left, keypoints, outputImage);
    cv::imwrite("sift_keypoints.jpg", outputImage);
    //imshow("结果图", outputImage);
    //cv::waitKey(0);


    // 定义点云使用的格式：这里用的是XYZRGB
    PointCloud<PointXYZRGB>::Ptr road_cloud(new PointCloud<PointXYZRGB>);

    // 根据视差和相机模型计算每一个点的三维坐标, 并添加到PCL点云中
    for (int v = 0; v < left.rows; v++)
    {
        for (int u = 0; u < left.cols; u++)
        {
            if (disparity.at<float>(v, u) <= 2.0 || disparity.at<float>(v, u) >= 96.0)
                continue;

            double depth = fx * b / (disparity.at<float>(v, u));
            pcl::PointXYZRGB p;
            p.x = depth * (u - cx) / fx;
            p.y = depth * (v - cy) / fy;
            p.z = depth;

            // 处理灰度图像
            uchar gray_value = left.at<uchar>(v, u); // 读取灰度值
            // 将灰度值映射到RGB通道
            p.r = gray_value; // 红色通道
            p.g = gray_value; // 绿色通道
            p.b = gray_value; // 蓝色通道

            /*for (const auto& kp : keypoints)
            {
                if (u == round(kp.pt.x) && v == round(kp.pt.y))
                {
                    p.r = 255; // 红色通道
                    p.g = 0; // 绿色通道
                    p.b = 0; // 蓝色通道


                }
            }*/




            road_cloud->points.push_back(p);
        }
    }
    // 缩放视差图以减小显示尺寸
    //cv::Mat resizedDisp;
    //cv::resize(disparity / 96.0, resizedDisp, cv::Size(), 0.2, 0.2);  // 缩小为原来的 50%
    //可视化视差图
    //cv::imshow("resizedDisp", resizedDisp);
    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);

    //可视化三维点云
    road_cloud->height = 1;
    road_cloud->width = road_cloud->points.size();
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(road_cloud);

    // 指定保存文件的路径
    std::string filename = "E:\\Cloud_Point_Prj\\PCL_POINT\\CMakeProject1\\output.pcd";

    // 保存点云到PCD文件
    if (pcl::io::savePCDFile(filename, *road_cloud) == -1) {
        PCL_ERROR("保存文件失败。\n");
        return -1;
    }

    std::cout << "点云已保存到文件: " << filename << std::endl;

    while (!viewer.wasStopped()) {}

    return 0;
}