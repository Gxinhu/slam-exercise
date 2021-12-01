#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
int main()
{
    Eigen::Quaterniond q1(0.35, 0.2, 0.3, 0.1);
    Eigen::Vector3d t1(0.3, 0.1, 0.1);
    Eigen::Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
    Eigen::Vector3d t2(-0.1, 0.5, 0.3);
    std::cout << "q1: " << q1.coeffs().transpose() << std::endl;
    // 直接使用四元数，需要先归一化
    q1.normalize();
    q2.normalize();
    std::cout << "q1: " << q1.coeffs().transpose() << std::endl;
    Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
    T1.rotate(q1);
    T1.pretranslate(t1);
    T2.rotate(q2);
    T2.pretranslate(t2);
    Eigen::Vector3d p(0.5, 0, 0.2);
    auto result = T2 * T1.inverse() * p;
    std::cout << "result: " << std::endl
              << result << std::endl;
}