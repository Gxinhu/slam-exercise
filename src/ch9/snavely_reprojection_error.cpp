#include "common.h"
#include "rotation.h"
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <iostream>
class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observered_x_(observed_x), observered_y_(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        T predition[2];
        CamprojectionWithDistortion(camera, point, predition);
        residuals[0] = predition[0] - T(observered_x_);
        residuals[1] = predition[1] - T(observered_y_);
        return true;
    }

    template <typename T>
    /**
     * @brief 
     * 
     * @param camera:  9d array [0-2]: angle-axis rotation, [3-5]


       translation , [6-8] camera parameter--focal length, second adn forth order radial

       distortion

       * @param point 3d world point coordinate 
     * @param prediction 2d

       preditoins with center
       of the image plane
     * @return true 
     * @return false


       * 
     */
    static inline bool CamprojectionWithDistortion(const T* camera, const T* point, T* prediction) {
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        T xp           = -p[0] / p[2];
        T yp           = -p[1] / p[2];
        const T& l1    = camera[7];
        const T& l2    = camera[8];
        T r2           = xp * xp + yp * yp;
        T distortion   = T(1.0) + r2 * (l1 + l2 * r2);
        const T& focal = camera[6];
        prediction[0]  = focal * distortion * xp;
        prediction[1]  = focal * distortion * yp;
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observered_x_;
    double observered_y_;
};
void SolveBA(BALProblem& bal_probelm);
int main() {
    std::string file_path = "/workspace/slam-exercise/src/ch9/problem-16-22106-pre.txt";
    BALProblem bal_problem(file_path);
    bal_problem.Normalize();
    bal_problem.WriteToPLYFile("initial.ply");
    bal_problem.Perturb(0.1, 0.5, 0.5);
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");
    return 0;
}

void SolveBA(BALProblem& bal_problem) {
    const int point_block_size  = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double* points              = bal_problem.mutable_points();
    double* caremas             = bal_problem.mutable_cameras();
    const double* observations  = bal_problem.observations();
    ceres::Problem problem;
    for (int i = 0; i <bal_problem.num_observations() ; ++i) {
        ceres::CostFunction* cost_function;
        cost_function =
            SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
        double* camera = caremas + camera_block_size * bal_problem.camera_index()[i];
        double* point  = points + point_block_size * bal_problem.point_index()[i];

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;
    std::cout << "Solveing ceres Ba ..." << std::endl;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_SCHUR; 
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
}