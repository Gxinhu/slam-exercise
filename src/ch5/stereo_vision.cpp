#include <pangolin/pangolin.h>
#include <unistd.h>

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>
const std::string LEFT_IMAGE = "./left.png";
const std::string RIGHT_IMAGE = "./right.png";
void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud);
int main(int argc, char **argv) {
    double fx = 718.856, fy = 718.856, cx = 601.1928, cy = 185.2157;
    // 基线
    double b = 0.573;
    cv::Mat left_image = cv::imread(LEFT_IMAGE, 0);
    cv::Mat right_image = cv::imread(RIGHT_IMAGE, 0);

    // 下面的魔法值是 SGBM 算法的神奇参数
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left_image, right_image, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> point_cloud;
    for (int v = 0; v < left_image.rows; ++v) {
        for (int u = 0; u < right_image.cols; ++u) {
            if (disparity.at<float>(v, u) <= 10.00 || disparity.at<float>(v, u) >= 96) {
                continue;
            }
            Eigen::Vector4d point(0, 0, 0, left_image.at<uchar>(v, u) / 255.0);
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;
            point_cloud.push_back(point);
        }
    }

    cv::imshow("left", left_image);
    cv::imshow("right", right_image);
    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    showPointCloud(point_cloud);
    return 0;
}
void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud) {
    if (pointcloud.empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p : pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);
    }
    return ;
}
