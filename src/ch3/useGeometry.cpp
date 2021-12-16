#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
int main(int argc, char** argv) {
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));
    std::cout.precision(3);
    std::cout << "rotation_matrix:\n" << rotation_vector.matrix() << std::endl;
    rotation_matrix = rotation_vector.toRotationMatrix();
    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d vector_rotated = rotation_matrix * v;
    std::cout << "(1,0,0) after rotation (by angle axis)= " << vector_rotated.transpose() << std::endl;
    vector_rotated = rotation_vector * v;
    std::cout << "(1,0,0) after rotation (by angle axis)= " << vector_rotated.transpose() << std::endl;
    Eigen::Quaterniond rotated_quaternion = Eigen::Quaterniond(rotation_vector);
    vector_rotated                        = rotated_quaternion * v;
    std::cout << "(1,0,0) after rotation (by quaternion)= " << vector_rotated.transpose() << std::endl;
    std::cout << "quaternion:\n" << rotated_quaternion.coeffs().transpose() << std::endl;
    auto original_quaternion_rotate =
        rotated_quaternion * Eigen::Quaterniond(0, 1, 0, 0) * rotated_quaternion.inverse();
    std::cout << "(1,0,0) after rotation (by quaternion)= " << original_quaternion_rotate.coeffs().transpose()
              << std::endl;

    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
    std::cout << "yaw pitch roll = " << euler_angles.transpose() << std::endl;

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1, 3, 5));
    std::cout << "Transform Matrix = \n" << T.matrix() << std::endl;
    Eigen::Vector3d v_transformed = T * v;
    std::cout << "(1,0,0) after transformation = " << v_transformed.transpose() << std::endl;
    Eigen::AngleAxisd vector(M_PI / 2, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotation(vector);
    std::cout << "向 Z 轴旋转 90 的旋转矩阵为" << rotation.matrix() << std::endl;
    return 0;
}