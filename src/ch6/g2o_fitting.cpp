// 比较复杂 等到看完全书之后，再来好好理解这个库
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }
    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update);
    }
    virtual bool read(std::istream &in) { return true; }
    virtual bool write(std::ostream &out) const { return true; }
};
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
    virtual void computeError() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }
    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }
    virtual bool read(std::istream &in) { return true; }
    virtual bool write(std::ostream &out) const { return true; }

   public:
    double _x;
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
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    for (int i = 0; i < N; ++i) {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    Eigen::Vector3d estimate = v->estimate();
    std::cout << "estimated values :" << estimate.transpose() << std::endl;
    return 0;
}