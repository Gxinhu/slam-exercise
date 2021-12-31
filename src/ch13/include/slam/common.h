
#ifndef CH13_COMMON_H_
#define CH13_COMMON_H_
#include <Eigen/Eigen>
#include <cstdint>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <sophus/se3.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <vector>
namespace slam {
const auto logger =
    spdlog::basic_logger_mt("basic_logger", "logs/basic-log.log");
}
#endif