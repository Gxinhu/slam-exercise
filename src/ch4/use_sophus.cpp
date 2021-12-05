#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <sophus/se3.hpp>

int main(int argc, char **argv) {
    Eigen::AngleAxisd rotation_vector = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();
    Eigen::Quaterniond quatuerion(rotation_matrix);

    Sophus::SO3d rotation_matrix_lie_group(rotation_matrix);
    Sophus::SO3d quatuerion_lie_group(quatuerion);

    std::cout << "rotation matrix: \n"
              << rotation_matrix << std::endl;
    std::cout << "SO(3) from rotation matrix:\n"
              << rotation_matrix_lie_group.matrix() << std::endl;
    std::cout << "SO(3) from quatuerion: \n"
              << quatuerion_lie_group.matrix() << std::endl;
    
}