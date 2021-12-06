#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <sophus/se3.hpp>

int main(int argc, char **argv) {
    Eigen::AngleAxisd rotation_vector = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();
    Eigen::Quaterniond quaternion(rotation_matrix);

    Sophus::SO3d rotation_matrix_lie_group(rotation_matrix);
    Sophus::SO3d quaternion_lie_group(quaternion);

    std::cout << "rotaion vector angle: "
              << rotation_vector.angle() << "\n"
              << "rotaion vector axis: "
              << rotation_vector.axis()
              << std::endl;
    std::cout << "rotation matrix: \n"
              << rotation_matrix << std::endl;
    std::cout << "SO(3) from rotation matrix:\n"
              << rotation_matrix_lie_group.matrix() << std::endl;
    std::cout << "SO(3) from quaternion: \n"
              << quaternion_lie_group.matrix() << std::endl;
    Eigen::Vector3d rotation_matrix_lie_algebra = rotation_matrix_lie_group.log();
    Eigen::Matrix3d rotation_matrix_lie_algebra_hat = Sophus::SO3d::hat(rotation_matrix_lie_algebra);
    Eigen::Vector3d rotation_matrix_lie_algebra_vee = Sophus::SO3d::vee(rotation_matrix_lie_algebra_hat);
    std::cout << "rotaion matrix lie algebra is : \n"
              << rotation_matrix_lie_algebra.transpose() << std::endl;
    std::cout << "rotaion matrix let algebra hat operation( return a skew symmetric matrix): \n"
              << rotation_matrix_lie_algebra_hat
              << std::endl;
    std::cout << "rotaion matrix let algebra hat'vee (same as original lie algebra): \n"
              << rotation_matrix_lie_algebra_vee.transpose()
              << std::endl;

    Eigen::Vector3d perturbation(1e-4, 0, 0);
    Sophus::SO3d rotation_matrix_perturb = Sophus::SO3d::exp(perturbation) * rotation_matrix_lie_group;
    std::cout << "perturbed rotation matrix lie = :\n"
              << rotation_matrix_perturb.matrix() << std::endl;
    Eigen::Vector3d translation_vector(1, 0, 0);
    Sophus::SE3d translation_matrix_lie_group(rotation_matrix, translation_vector);
    Sophus::SE3d translation_quaternion_lie_group(quaternion, translation_vector);
    std::cout << "SE3 from rotation matrix and translation vector: \n"
              << translation_matrix_lie_group.matrix() << std::endl;
    std::cout << "SE3 from quaternion and translation vector: \n"
              << translation_quaternion_lie_group.matrix() << std::endl;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d translation_matrix_lie_algebra = translation_matrix_lie_group.log();
    std::cout << "translation matrix lie algebra:\n"
              << translation_matrix_lie_algebra.transpose() << std::endl;
    std::cout << "tanslation matrix lie algebra hat:\n"
              << Sophus::SE3d::hat(translation_matrix_lie_algebra)
              << std::endl;
    std::cout << "translation matrix lie algebra hat vee: \n"
              << Sophus::SE3d::vee(Sophus::SE3d::hat(translation_matrix_lie_algebra))
              << std::endl;
    Vector6d perturb_translation;
    perturb_translation.setZero();
    perturb_translation(0, 0) = 1e-4;
    Sophus::SE3d perturb_lie_group = Sophus::SE3d::exp(perturb_translation) * translation_matrix_lie_group;
    std::cout << "perturb lie group = \n"
              << perturb_lie_group.matrix() << std::endl;
    std::cout << "perturb vector = \n"
              << perturb_translation.transpose() << std::endl;
    return 0;
}