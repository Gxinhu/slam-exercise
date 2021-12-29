#include <Eigen/Eigen>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Transform.h>
#include <argparse/argparse.hpp>
#include <array>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
#include <octomap/octomap_types.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <sys/types.h>
#include <vector>

using Point              = pcl::PointXYZRGB;
using PointCloud         = pcl::PointCloud<Point>;
using Channel            = cv::Vec<std::uint8_t, 3>;
const double kCx         = 319.5;
const double kCy         = 239.5;
const double kFx         = 481.2;
const double kFy         = -480.0;
const double kDepthScale = 5000.0;
/**
 * @brief Create a Point Cloud Map
 *
 * @param color_images 彩色图像集
 * @param poses 位姿集合
 * @param depth_images 深度图像集
 */
void createPointCloud(const std::vector<cv::Mat>& color_images,
    const std::vector<Eigen::Isometry3d>& poses, const std::vector<cv::Mat>& depth_images);
/**
 * @brief Create a Octomap
 *
 * @param color_images
 * @param poses
 * @param depth_images
 */
void createOctomap(const std::vector<cv::Mat>& color_images,
    const std::vector<Eigen::Isometry3d>& poses, const std::vector<cv::Mat>& depth_images);
int main(int argc, char** argv) {

    argparse::ArgumentParser program("point_cloud");
    program.add_argument("--path")
        .required()
        .help("dateset path")
        .default_value(std::string("/workspace/slam-exercise/src/ch12/data"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto path      = program.get<std::string>("--path");
    auto file_path = fmt::format("{}/pose.txt", path);
    std::ifstream fin(file_path);
    if (!fin) {
        std::cerr << "can not find pose file" << std::endl;
        return -1;
    }
    std::vector<cv::Mat> color_images;
    std::vector<cv::Mat> depth_images;
    std::vector<Eigen::Isometry3d> poses;

    for (int i = 0; i < 5; ++i) {
        auto image_path = fmt::format("{}/color/{}.png", path, i + 1);
        auto depth_path = fmt::format("{}/depth/{}.png", path, i + 1);
        color_images.push_back(cv::imread(image_path));
        depth_images.push_back(cv::imread(depth_path, cv::IMREAD_UNCHANGED));
        std::array<double, 7> data;
        for (int i = 0; i < 7; ++i) {
            fin >> data[i];
        }
        Eigen::Quaterniond quaternion(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d transform(quaternion);
        transform.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(transform);
    }
    createOctomap(color_images, poses, depth_images);
    createPointCloud(color_images, poses, depth_images);
    return 0;
}

void createPointCloud(const std::vector<cv::Mat>& color_images,
    const std::vector<Eigen::Isometry3d>& poses, const std::vector<cv::Mat>& depth_images) {
    fmt::print("transfrom images to point cloud...\n");
    PointCloud::Ptr point_cloud(new PointCloud);
    for (int i = 0; i < 5; i++) {
        PointCloud::Ptr current(new PointCloud);
        auto color_image       = color_images[i];
        auto depth_image       = depth_images[i];
        Eigen::Isometry3d pose = poses[i];
        for (int v = 0; v < color_image.rows; ++v) {
            for (int u = 0; u < color_image.cols; ++u) {
                unsigned int depth = depth_image.at<std::uint16_t>(v, u);
                if (depth == 0) {
                    continue;
                }
                Eigen::Vector3d point;
                point[2]                    = double(depth) / kDepthScale;
                point[0]                    = (u - kCx) * point[2] / kFx;
                point[1]                    = (v - kCy) * point[2] / kFy;
                Eigen::Vector3d world_point = pose * point;
                Point p;
                p.x = world_point[0];
                p.y = world_point[1];
                p.z = world_point[2];
                p.r = color_image.at<Channel>(v, u)[0];
                p.g = color_image.at<Channel>(v, u)[1];
                p.b = color_image.at<Channel>(v, u)[2];
                current->points.push_back(p);
            }
        }
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<Point> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*point_cloud) += *tmp;
    }
    point_cloud->is_dense = false;
    fmt::print("point cloud size:{}\n", point_cloud->size());

    pcl::VoxelGrid<Point> voxel_filter;
    double resolution = 0.03;
    voxel_filter.setLeafSize(resolution, resolution, resolution);
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(point_cloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*point_cloud);

    fmt::print("point cloud size after filter:{}\n", point_cloud->size());
    pcl::io::savePCDFile("map.pcd", *point_cloud);
}


void createOctomap(const std::vector<cv::Mat>& color_images,
    const std::vector<Eigen::Isometry3d>& poses, const std::vector<cv::Mat>& depth_images) {
    octomap::OcTree tree(0.01);
    fmt::print("transfrom images to octomap...\n");
    for (int i = 0; i < 5; i++) {
        cv::Mat color_image    = color_images[i];
        cv::Mat depth_image    = depth_images[i];
        Eigen::Isometry3d pose = poses[i];
        octomap::Pointcloud point_cloud;
        for (int v = 0; v < color_image.rows; ++v) {
            for (int u = 0; u < color_image.cols; ++u) {
                unsigned int d = depth_image.at<std::uint16_t>(v, u);
                if (d == 0) {
                    continue;
                }
                Eigen::Vector3d point;
                point[2]                    = double(d) / kDepthScale;
                point[0]                    = (u - kCx) * point[2] / kFx;
                point[1]                    = (v - kCy) * point[2] / kFy;
                Eigen::Vector3d world_point = pose * point;
                point_cloud.push_back(world_point[0], world_point[1], world_point[2]);
            }
        }
        tree.insertPointCloud(point_cloud, octomap::point3d(pose(0, 3), pose(1, 3), pose(2, 3)));
    }
    tree.updateInnerOccupancy();
    tree.writeBinary("octomap.bt");
    fmt::print("save octomap done\n");
}