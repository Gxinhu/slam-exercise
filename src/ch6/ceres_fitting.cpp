#include <ceres/ceres.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
// In C++ the only difference between a class and a struct is that members and base classes are private by default in classes,
// whereas they are public by default in structs.

struct Cost_Function {
    Cost_Function(double x, double y) : _x(x), _y(x) {}
    template <typename T>
    bool operator()(const T *const abc, T *residual)
        const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }
    const double _x, _y;
};
int main() {
    double ar = 1.0, br = 2.0, cr = 1.0;
    // estimation value
    double ae = 2.0, be = -1.0, ce = 5.0;
    const int N = 100;
    double w_sigma = 1.0;
    double inverse_sigma = 1.0 / w_sigma;
    cv::RNG random_generator;
    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + random_generator.gaussian(w_sigma * w_sigma));
    }
    double abc[3] = {ae, be, ce};
    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        ceres::CostFunction *const_function = new ceres::AutoDiffCostFunction<Cost_Function, 1, 3>(new Cost_Function(x_data[i], y_data[i]));
        problem.AddResidualBlock(const_function, nullptr, abc);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    auto time1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    auto time2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(time2 - time1);
    std::cout << "Time is " << duration.count() << std::endl;
    std::cout << summary.BriefReport() << std::endl;
    return 0;
}