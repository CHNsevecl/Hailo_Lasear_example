#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Myhailo.hpp"

int main(){
    setenv("DISPLAY", ":0", 1);
    const std::string hef_path = "yolov8s_for_BLasear_V1_1.hef";
    constexpr size_t expected_input = 640*640*3;

    std::optional<HailoContext> hailo_context = Hailo_init(hef_path);
    if (!hailo_context) {
        std::cerr << "Error: Failed to initialize Hailo context." << std::endl;
        return -1;
    }

    std::string pipeline = 
        "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/imx708@1a ! "
        "video/x-raw, format=NV12, width=640, height=480, framerate=60/1 ! "
        "appsink drop=true max-buffers=1 sync=false";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return -1;
    }

    const int target_w = 640;
    const int target_h = 640;
    const std::vector<std::string> class_names = {"Blue_Lasear"};
    
    while (true){
        cv::Mat NV12_frame;
        if (!cap.read(NV12_frame)) {
            std::cerr << "Error: Could not read frame from video stream." << std::endl;
            break;
        }

        cv::Mat BGR_frame;
        cv::cvtColor(NV12_frame, BGR_frame, cv::COLOR_YUV2BGR_NV12);

        cv::Mat display_frame = Stream_process(BGR_frame, target_w, target_h);
        
        std::optional<std::vector<Detection>> detections_opt = ParseDetections(*hailo_context, target_w, target_h, display_frame, class_names);
        cv::cvtColor(display_frame, display_frame, cv::COLOR_RGB2BGR); // 将RGB转回BGR以便OpenCV显示

        if (!detections_opt) {
            std::cerr << "Error: Failed to parse detections." << std::endl;
            break;
        }
        const auto& detections = *detections_opt;
        for (const auto& det : detections) {
            if(det.score < 0.5) { // 过滤掉低置信度的检测结果
                continue;
            }
            if(det.label == "Blue_Lasear") {
                cv::Point center((det.upper.x + det.lower.x) / 2, (det.upper.y + det.lower.y) / 2);
                cv::circle(display_frame, center, 2, cv::Scalar(255, 0, 0), -1); // 在检测到的激光点位置画一个绿色圆点
                cv::putText(display_frame,                                      // 图像
                    det.label + " " + cv::format("%.2f", det.score),                    // 文本内容
                    cv::Point(center.x - 20, std::max(0, center.y - 10)),       // 文本位置
                    cv::FONT_HERSHEY_SIMPLEX,                                   // 字体类型
                    0.5,                                                        // 字体大小
                    cv::Scalar(0, 255, 0),                                      // 颜色（BGR）
                    1                                                           // 线条粗细
                );
            }
        }
       
        cv::imshow("Hailo Inference", display_frame);
        if (cv::waitKey(1) == 27) { // Exit on 'ESC' key press
            break;
        }
    }
    
}