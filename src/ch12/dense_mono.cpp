#include <Eigen/Eigen>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/ReturnByValue.h>
#include <argparse/argparse.hpp>
#include <cmath>
#include <cstddef>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <optional>
#include <sophus/se3.hpp>
#include <stdexcept>
#include <string>
#include <vector>

const int kBoarder       = 20;
const int kWidth         = 640;
const int kHeight        = 480;
const double kFx         = 481.2F;
const double kFy         = -480.0F;
const double kCx         = 319.5F;
const double kCy         = 239.5F;
const int kNccWindowSize = 3;
const int kNccArea       = (2 * kNccWindowSize + 1) * (2 * kNccWindowSize + 1);
const double kMinCov     = 0.1;
const double kMaxCov     = 10;
/**
 * @brief 根据图像更新深度估计
 *
 * @param ref 参考图像
 * @param curr  当前图像
 * @param T_c_r  参考图像到到当前图像的位姿
 * @param depth  深度
 * @param depth_cov2  深度方差
 * @return true
 * @return false
 */
void update(const cv::Mat& reference, const cv::Mat& current, const Sophus::SE3d& T_c_r,
    cv::Mat& depth, cv::Mat& depth_variance);
/**
 * @brief 极线搜索
 *
 * @param reference  参考图像
 * @param current  当前图像
 * @param T_c_r  参考图像到当前图像的位姿
 * @param reference_point  参考图像点的位置
 * @param depth_mean  深度均值
 * @param depth_variance  深度方差
 * @param current_point  当前点的位置
 * @param epipolar_direction  极线方向
 * @return true
 * @return false
 */
bool epipolarSearch(const cv::Mat& reference, const cv::Mat& current, const Sophus::SE3d& T_c_r,
    const Eigen::Vector2d& reference_point, const double& depth_mean, const double& depth_variance,
    Eigen::Vector2d& current_point, Eigen::Vector2d& epipolar_direction);

/**
 * @brief
 *
 * @param reference_point
 * @param current_point
 * @param T_c_r
 * @param epipolar_direction
 * @param depth
 * @param depth_variance
 * @return true
 * @return false
 */
bool updateDepthFilter(const Eigen::Vector2d& reference_point, const Eigen::Vector2d& current_point,
    const Sophus::SE3d& T_c_r, const Eigen::Vector2d& epipolar_direction, cv::Mat& depth,
    cv::Mat& depth_variance);
/**
 * @brief
 *
 * @param reference
 * @param current
 * @param reference_point
 * @param current_point
 * @return double
 */
double NCC(const cv::Mat& reference, const cv::Mat& current, const Eigen::Vector2d& reference_point,
    const Eigen::Vector2d& current_point);
/**
 * @brief Get the Bilinear Interpolated Value object
 *
 * @param image  opencv matrix
 * @param point  Eigen vector represents position in image
 * @return double 经过双线性插值之后的灰度值
 */
inline double getBilinearInterpolatedValue(const cv::Mat& image, const Eigen::Vector2d& point) {
    uchar* d  = &image.data[int(point(1, 0) * image.step + int(point(0, 0)))];
    double xx = point(0, 0) - floor(point(0, 0));
    double yy = point(1, 0) - floor(point(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0])
               + xx * (1 - yy)
                     * double(d[1] + (1 - xx) * yy * double(d[image.step])
                              + xx * yy * double(d[image.step + 1])))
         / 255.0;
};
/**
 * @brief
 *
 * @param path
 * @param color_image_files
 * @param poses_world2camera
 * @param reference_depth
 * @return true
 * @return false
 */

bool readDatasetFiles(const std::string& path, std::vector<std::string>& color_image_files,
    std::vector<Sophus::SE3d>& poses_world2camera, cv::Mat& ref_depth);

inline Eigen::Vector3d pixel2camera(const Eigen::Vector2d& pixel) {
    return Eigen::Vector3d((pixel(0, 0) - kCx) / kFx, (pixel(1, 0) - kCy) / kFy, 1);
}

inline Eigen::Vector2d camera2pixel(const Eigen::Vector3d& camera_point) {
    return Eigen::Vector2d(camera_point(0, 0) * kFx / camera_point(2, 0) + kCx,
        camera_point(1, 0) * kFy / camera_point(2, 0) + kCy);
}

inline bool inside(const Eigen::Vector2d& pt) {
    return pt(0, 0) >= kBoarder && pt(1, 0) >= kBoarder && pt(0, 0) + kBoarder < kWidth
        && pt(1, 0) + kBoarder <= kHeight;
}

void evaludateDepth(const cv::Mat& depth_truth, const cv::Mat& depth_estimate);

void plotDepth(const cv::Mat& depth_truth, const cv::Mat& depth_estimate);
int main(int argc, char** argv) {
    argparse::ArgumentParser program("dense_mono");
    program.add_argument("--path")
        .required()
        .help("dateset path")
        .default_value(std::string("/workspace/slam-exercise/src/ch12/test_data"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto path = program.get<std::string>("--path");
    std::vector<std::string> color_image_files;
    std::vector<Sophus::SE3d> poses_world2camera;
    cv::Mat reference_depth;
    bool is_read = readDatasetFiles(path, color_image_files, poses_world2camera, reference_depth);
    if (!is_read) {
        std::cout << "Reading image files failed" << std::endl;
        return -1;
    }
    std::cout << "read total " << color_image_files.size() << " files." << std::endl;
    cv::Mat reference = cv::imread(color_image_files[0], cv::IMREAD_GRAYSCALE);
    Sophus::SE3d pose_reference_world2camera = poses_world2camera[0];
    const double init_depth                  = 3.0;
    const double init_variance               = 3.0;

    cv::Mat depth(kHeight, kWidth, CV_64F, init_depth);
    cv::Mat depth_variance(kHeight, kWidth, CV_64F, init_variance);

    for (int index = 1; index < color_image_files.size(); ++index) {
        fmt::print("----loop {} ----\n", index);
        cv::Mat current = cv::imread(color_image_files[index], cv::IMREAD_GRAYSCALE);
        if (current.data == nullptr) {
            continue;
        }
        Sophus::SE3d pose_current_world2camera = poses_world2camera[index];
        Sophus::SE3d pose_currnet2reference =
            pose_current_world2camera.inverse() * pose_reference_world2camera;
        update(reference, current, pose_currnet2reference, depth, depth_variance);
        evaludateDepth(reference_depth, depth);
        plotDepth(reference_depth, depth);
        cv::imshow("image", current);
    }


    return 0;
}

void plotDepth(const cv::Mat& depth_truth, const cv::Mat& depth_estimate) {
    cv::imshow("depth_truth", depth_truth * 0.4);
    cv::imshow("depth_estimate", depth_estimate * 0.4);
    cv::imshow("depth_error", depth_truth - depth_estimate);
    cv::waitKey(1);
}
void update(const cv::Mat& reference, const cv::Mat& current, const Sophus::SE3d& T_c_r,
    cv::Mat& depth, cv::Mat& depth_variance) {
    for (int x = kBoarder; x < kWidth - kBoarder; ++x) {
        for (int y = kBoarder; y < kHeight; ++y) {
            if (depth_variance.at<double>(y, x) < kMinCov
                || depth_variance.at<double>(y, x) > kMaxCov) {
                continue;
            }
            auto ff = depth.ptr<double>(y)[x];
            Eigen::Vector2d current_point;
            Eigen::Vector2d epipolar_direction;
            bool ret = epipolarSearch(reference, current, T_c_r, Eigen::Vector2d(x, y),
                depth.at<double>(y, x), depth_variance.at<double>(y, x), current_point,
                epipolar_direction);
            if (!ret) {
                continue;
            }
            updateDepthFilter(Eigen::Vector2d(x, y), current_point, T_c_r, epipolar_direction,
                depth, depth_variance);
        }
    }
}

bool epipolarSearch(const cv::Mat& reference, const cv::Mat& current, const Sophus::SE3d& T_c_r,
    const Eigen::Vector2d& reference_point, const double& depth_mean, const double& depth_variance,
    Eigen::Vector2d& current_point, Eigen::Vector2d& epipolar_direction) {
    Eigen::Vector3d reference_no_depth = pixel2camera(reference_point);
    reference_no_depth.normalize();
    // 参考图像中，按照均值深度所计算的空间点
    Eigen::Vector3d reference_p = reference_no_depth * depth_mean;
    // 按照均值深度计算空间点投影到当前图像的像素
    Eigen::Vector2d pixel_mean_current = camera2pixel(T_c_r * reference_p);
    double depth_min                   = depth_mean - 3 * depth_variance;
    double depth_max                   = depth_mean + 3 * depth_variance;
    const double minest_depth          = 0.1;
    if (depth_min < minest_depth) {
        depth_min = minest_depth;
    }

    Eigen::Vector2d pixel_min_current = camera2pixel(T_c_r * (reference_no_depth * depth_min));
    Eigen::Vector2d pixel_max_current = camera2pixel(T_c_r * (reference_no_depth * depth_max));
    Eigen::Vector2d epipolar_line     = pixel_max_current - pixel_min_current;
    epipolar_direction                = epipolar_line;
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();
    if (half_length > 100) {
        half_length = 100;
    }
    double best_ncc = -1.0;
    Eigen::Vector2d best_pixel_current;
    for (double l = -half_length; l <= half_length; l += 0.7) {
        Eigen::Vector2d current_pixel = pixel_max_current + l * epipolar_direction;
        if (!inside(current_pixel)) {
            continue;
        }
        double ncc = NCC(reference, current, reference_point, current_pixel);
        if (ncc > best_ncc) {
            best_ncc           = ncc;
            best_pixel_current = current_pixel;
        }
    }
    const double ncc_thread = 0.85F;
    if (best_ncc < ncc_thread) {
        return false;
    }
    current_point = best_pixel_current;
    return true;
}

double NCC(const cv::Mat& reference, const cv::Mat& current, const Eigen::Vector2d& reference_point,
    const Eigen::Vector2d& current_point) {
    double reference_mean = 0;
    double current_mean   = 0;
    std::vector<double> reference_values;
    std::vector<double> current_values;
    for (int x = -kNccWindowSize; x <= kNccWindowSize; ++x) {
        for (int y = -kNccWindowSize; y <= kNccWindowSize; ++y) {
            double reference_value = double(reference.at<uchar>(int(y + reference_point(1, 0)),
                                         int(x + reference_point(0, 0))))
                                   / 255.0;
            reference_mean += reference_value;
            double current_value =
                getBilinearInterpolatedValue(current, current_point + Eigen::Vector2d(x, y));
            current_mean += current_value;
            current_values.push_back(current_value);
            reference_values.push_back(reference_value);
        }
    }
    reference_mean /= kNccArea;
    current_mean /= kNccArea;
    double numerator   = 0;
    double demoniator1 = 0;
    double demoniator2 = 0;
    for (int i = 0; i < reference_values.size(); ++i) {
        double n = (reference_values[i] - reference_mean) * (current_values[i] - current_mean);
        numerator += n;
        demoniator1 +=
            (reference_values[i] - reference_mean) * (reference_values[i] - reference_mean);
        demoniator2 += (current_values[i] - current_mean) * (current_values[i] - current_mean);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

bool updateDepthFilter(const Eigen::Vector2d& reference_point, const Eigen::Vector2d& current_point,
    const Sophus::SE3d& T_c_r, const Eigen::Vector2d& epipolar_direction, cv::Mat& depth,
    cv::Mat& depth_variance) {
    // 匹配点 三角化 求深度
    Sophus::SE3d pose_r_c              = T_c_r.inverse();
    Eigen::Vector3d reference_no_depth = pixel2camera(reference_point);
    reference_no_depth.normalize();
    Eigen::Vector3d current_no_depth = pixel2camera(current_point);
    current_no_depth.normalize();
    Eigen::Vector3d translation = pose_r_c.translation();
    Eigen::Vector3d x2          = T_c_r.so3() * current_no_depth;
    // 7.5 节说的是用最小二乘法，而这里是直接求解
    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Eigen::Vector3d t  = pose_r_c.translation();
    Eigen::Vector3d f2 = pose_r_c.so3() * current_no_depth;
    Eigen::Vector2d b  = Eigen::Vector2d(t.dot(reference_no_depth), t.dot(f2));
    Eigen::Matrix2d A;
    A(0, 0)                    = reference_no_depth.dot(reference_no_depth);
    A(0, 1)                    = -reference_no_depth.dot(f2);
    A(1, 0)                    = -A(0, 1);
    A(1, 1)                    = -f2.dot(f2);
    Eigen::Vector2d ans        = A.inverse() * b;
    Eigen::Vector3d xm         = ans[0] * reference_no_depth; // ref 侧的结果
    Eigen::Vector3d xn         = t + ans[1] * f2; // cur 结果
    Eigen::Vector3d p_estimate = (xm + xn) / 2.0; // P的位置，取两者的平均
    double depth_estimation    = p_estimate.norm(); // 深度值

    // 计算不确定性（以一个像素为误差）
    Eigen::Vector3d p            = reference_no_depth * depth_estimation;
    Eigen::Vector3d a            = p - t;
    double t_norm                = t.norm();
    double a_norm                = a.norm();
    double alpha                 = acos(reference_no_depth.dot(t) / t_norm);
    double beta                  = acos(-a.dot(t) / (a_norm * t_norm));
    Eigen::Vector3d f_curr_prime = pixel2camera(current_point + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma      = M_PI - alpha - beta_prime;
    double p_prime    = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov      = p_prime - depth_estimation;
    double d_cov2     = d_cov * d_cov;

    // 高斯融合
    double mu = depth.ptr<double>(int(reference_no_depth(1, 0)))[int(reference_no_depth(0, 0))];
    double sigma2 =
        depth_variance.ptr<double>(int(reference_point(1, 0)))[int(reference_point(0, 0))];

    double mu_fuse     = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(reference_point(1, 0)))[int(reference_point(0, 0))] = mu_fuse;
    depth_variance.ptr<double>(int(reference_point(1, 0)))[int(reference_point(0, 0))] =
        sigma_fuse2;

    return true;
}

void evaludateDepth(const cv::Mat& depth_truth, const cv::Mat& depth_estimate) {
    double ave_depth_error    = 0; // 平均误差
    double ave_depth_error_sq = 0; // 平方误差
    int cnt_depth_data        = 0;
    for (int y = kBoarder; y < depth_truth.rows - kBoarder; y++) {
        for (int x = kBoarder; x < depth_truth.cols - kBoarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    std::cout << "Average squared error = " << ave_depth_error_sq
              << ", average error: " << ave_depth_error << std::endl;
}

bool readDatasetFiles(const std::string& path, std::vector<std::string>& color_image_files,
    std::vector<Sophus::SE3d>& poses, cv::Mat& ref_depth) {
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin)
        return false;

    while (!fin.eof()) {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        std::string image;
        fin >> image;
        double data[7];
        for (double& d : data)
            fin >> d;

        color_image_files.push_back(path + std::string("/images/") + image);
        poses.push_back(Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
            Eigen::Vector3d(data[0], data[1], data[2])));
        if (!fin.good())
            break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(kHeight, kWidth, CV_64F);
    if (!fin)
        return false;
    for (int y = 0; y < kHeight; y++) {
        for (int x = 0; x < kWidth; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }
    }
    return true;
}