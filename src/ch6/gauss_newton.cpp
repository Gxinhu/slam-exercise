#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>

#include <opencv2/opencv.hpp>
int main() {
    // real value
    const double ar = 1.0;
    const double br = 2.0;
    const double cr = 1.0;
    // estimation value
    double ae                  = 2.0;
    double be                  = -1.0;
    double ce                  = 5.0;
    const int n                = 100;
    const double w_sigma       = 1.0;
    const double inverse_sigma = 1.0 / w_sigma;
    cv::RNG random_generator;
    std::vector<double> x_data;
    std::vector<double> y_data;
    std::vector<double> real_y;
    std::vector<double> estimate_y;
    for (int i = 0; i < n; ++i) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + random_generator.gaussian(w_sigma * w_sigma));
        real_y.push_back(exp(ar * x * x + br * x + cr));
    }
    const int iterations = 100;
    double cost;
    double last_cost = 0;

    auto time1 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        cost              = 0;
        for (int i = 0; i < n; ++i) {
            double xi    = x_data[i];
            double yi    = y_data[i];
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Eigen::Vector3d J;
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);
            J[2] = -exp(ae * xi * xi + be * xi + ce);
            H += inverse_sigma * inverse_sigma * J * J.transpose();
            b += -inverse_sigma * inverse_sigma * error * J;
            cost += error * error;
        }
        Eigen::Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            std::cout << "result is nan" << std::endl;
            break;
        }
        if (iter > 0 && cost >= last_cost) {
            fmt::print("cost: {} ,last cost:{}\n", cost, last_cost);
        }
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        last_cost = cost;
        fmt::print("total cost: {:.4f}\t update: {} \t estimated "
                   "params:{:.4f},{:.4f},{:.4f}\n",
            cost, dx.transpose(), ae, be, ce);
    }
    auto time2    = std::chrono::steady_clock::now();
    auto difftime = std::chrono::duration_cast<std::chrono::duration<double>>(time2 - time1);
    std::cout << " estimation time is " << difftime.count() << " ms" << std::endl;
    for (int i = 0; i < n; ++i) {
        double x = i / 100.0;
        estimate_y.push_back(exp(ae * x * x + be * x + ce));
    }
    matplot::plot(x_data, estimate_y, "r", x_data, real_y, "b", x_data, y_data, "y.");
    matplot::show();

    return 0;
}