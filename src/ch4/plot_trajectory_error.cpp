#include <pangolin/pangolin.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <sophus/se3.hpp>
const std::string GROUND_TRUTH_FILE = "./examples/groundtruth.txt";
const std::string ESTIMATION_FILE = "./examples/estimated.txt";

// 这个地方是 Eigen 本身的问题，在使用 STL 容器的时候 Eigen要进行一些特殊操作，详情见 https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
void PlotTrajectory(const TrajectoryType &ground_truth, const TrajectoryType &estimation);
TrajectoryType ReadTrajectory(const std::string &path);
int main() {
    TrajectoryType ground_truth = ReadTrajectory(GROUND_TRUTH_FILE);
    TrajectoryType estimation = ReadTrajectory(ESTIMATION_FILE);
    assert(!ground_truth.empty() && !estimation.empty());
    assert(ground_truth.size() == estimation.size());
    double root_mean_squared_error = 0;
    for (int i = 0; i < estimation.size(); ++i) {
        Sophus::SE3d p1 = estimation[i], p2 = ground_truth[i];
        double error = (p2.inverse() * p1).log().norm();
        root_mean_squared_error += error;
    }
    root_mean_squared_error = root_mean_squared_error / double(estimation.size());
    root_mean_squared_error = sqrt(root_mean_squared_error);
    std::cout << "Root Mean Squared Error is " << root_mean_squared_error << std::endl;

    double average_translation_error = 0;
    for (int i = 0; i < estimation.size(); ++i) {
        Sophus::SE3d p1 = estimation[i], p2 = ground_truth[i];
        Eigen::Matrix4d translation_matrix = (p2.inverse() * p1).matrix();
        double tx = translation_matrix(0, 3), ty = translation_matrix(1, 3), tz = translation_matrix(2, 3);
        Eigen::Vector3d translation_vector(tx, ty, tz);
        double error = translation_vector.norm();
        average_translation_error += error;
    }
    average_translation_error = average_translation_error / double(estimation.size());
    average_translation_error = sqrt(average_translation_error);
    std::cout << "Average Translation Error is " << average_translation_error << std::endl;

    PlotTrajectory(ground_truth, estimation);
    return 0;
}
TrajectoryType ReadTrajectory(const std::string &path) {
    std::ifstream fin(path);
    TrajectoryType trajectory;
    if (!fin) {
        std::cerr << "Trajectory " << path << " not find" << std::endl;
        return trajectory;
    }
    while (!fin.eof()) {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3d quaternion_lie_group(Eigen::Quaterniond(qx, qy, qz, qw), Eigen::Vector3d(tx, ty, tz));
        trajectory.push_back(quaternion_lie_group);
    }
    return trajectory;
}
void PlotTrajectory(const TrajectoryType &ground_truth, const TrajectoryType &estimation) {
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);
        // 画出连线
        for (size_t i = 0; i < ground_truth.size(); i++) {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = ground_truth[i], p2 = ground_truth[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for (size_t i = 0; i < estimation.size(); i++) {
            glColor3f(1.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = estimation[i], p2 = estimation[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);  // sleep 5 ms
    }
}