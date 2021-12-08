#include <fmt/core.h>
#include <pangolin/pangolin.h>
#include <unistd.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

const std::string FILE_PATH = "/workspace/slam-exercise/src/ch5/rgbd";
typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
void showPointCloud(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);
int main() {
    std::vector<cv::Mat> color_images, depth_images;
    TrajectoryType poses;
    std::ifstream fin("/workspace/slam-exercise/src/ch5/rgbd/pose.txt");
    if (fin.fail()) {
        std::cerr << "Can not find pose file!" << std::endl;
        return 0;
    }
    for (int i = 0; i < 5; ++i) {
        cv::Mat color_matrix = cv::imread(fmt::format("{}/color/{}.png", FILE_PATH, i + 1));
        cv::Mat depth_matrix = cv::imread(fmt::format("{}/depth/{}.pgm", FILE_PATH, i + 1), -1);
        color_images.push_back(color_matrix);
        depth_images.push_back(depth_matrix);
        double data[7] = {0};
        for (auto &d : data) {
            fin >> d;
        }
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]), Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depth_scale = 1000.0;
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> point_cloud;
    point_cloud.reserve(1000000);
    for (int i = 0; i < 5; i++) {
        auto color_image = color_images[i];
        auto depth_image = depth_images[i];
        Sophus::SE3d T = poses[i];
        for (int v = 0; v < color_image.rows; v++) {
            for (int u = 0; u < color_image.cols; u++) {
                auto d = depth_image.at<unsigned short>(v, u);
                if (d == 0) {
                    continue;
                }
                Eigen::Vector3d point;
                point[2] = (double)d / depth_scale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d point_world = T * point;
                Vector6d p;
                p.head<3>() = point_world;
                // step means numbers of bytes each matrix row occupies.
                p[5] = color_image.data[v * color_image.step + u * color_image.channels()];
                p[4] = color_image.data[v * color_image.step + u * color_image.channels() + 1];
                p[3] = color_image.data[v * color_image.step + u * color_image.channels() + 2];
                point_cloud.push_back(p);
            }
        }
    }
    fmt::print("total point is {}\n", point_cloud.size());
    showPointCloud(point_cloud);
    return 0;
}

void showPointCloud(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {
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
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);  // sleep 5 ms
    }
    return;
}