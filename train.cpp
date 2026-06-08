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
        "video/x-raw, format=NV12, width=640, height=480, framerate=30/1 ! "
        "appsink drop=true max-buffers=1 sync=false";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return -1;
    }

    cv::namedWindow("Hailo Inference", cv::WINDOW_AUTOSIZE);

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

        //取缩放比最小的，确保缩放后的图像能够处于目标尺寸内
        const float scale = std::min(static_cast<float>(target_w) / BGR_frame.cols, static_cast<float>(target_h) / BGR_frame.rows);

        const int new_w = static_cast<int>(BGR_frame.cols * scale);
		const int new_h = static_cast<int>(BGR_frame.rows * scale);
		const int x_offset = (target_w - new_w) / 2; // 水平方向偏移
		const int y_offset = (target_h - new_h) / 2; // 垂直方向偏移

        cv::resize(BGR_frame, BGR_frame, cv::Size(new_w, new_h));

        cv::Mat letterbox_bgr(target_h, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
        BGR_frame.copyTo(letterbox_bgr(cv::Rect(x_offset, y_offset, new_w, new_h)));

        cv::Mat RGB_frame;
        cv::cvtColor(letterbox_bgr, RGB_frame, cv::COLOR_BGR2RGB);
        
        
        if (!RGB_frame.isContinuous()) {//copyTo、Rect等操作可能导致图像数据不连续，克隆一份确保数据连续
			RGB_frame = RGB_frame.clone();
		}
        //流数据处理完毕

        //流数据放入输入内存

        // input_buffers[0].size()                          // 模型输入缓冲区大小（字节）恰好是yolov8s,只有一个输入
        // resized_rgb.total()                              // 图像像素总数（宽×高）
        // resized_rgb.elemSize()                           // 每个像素的字节数
        // resized_rgb.total() * resized_rgb.elemSize()     // 图像数据总大小（字节）
        if (hailo_context->input_buffer[0].size() != RGB_frame.total() * RGB_frame.elemSize()) {
			std::cerr << "错误: 摄像头预处理后的输入大小与模型输入不一致" << std::endl;
			break;
		}

        std::memcpy(&hailo_context->input_buffer[0][0], RGB_frame.data, hailo_context->input_buffer[0].size());
        // 这两个是等价的,都是获取第一个元素的指针地址
        // hailo_context->input_buffer[0].data()
        // &hailo_context->input_buffer[0][0]

        auto run_status = hailo_context->configured_infer_model.run(hailo_context->bindings, std::chrono::milliseconds(1000));
        if (run_status != HAILO_SUCCESS) {
			std::cerr << "错误: 推理执行失败, status=" << run_status << std::endl;
			continue;
		}

        cv::Mat display_frame = letterbox_bgr.clone();
        letterbox_bgr.release(); // 释放中间变量占用的内存
        const float *detections = reinterpret_cast<const float*>(hailo_context->output_buffer[0].data()); // 将输出缓冲按float数组解析
        const size_t detection_float_count = hailo_context->output_buffer[0].size() / sizeof(float); // 输出里一共有多少个float
        size_t detection_offset = 0; // 解析游标，每读取一个字段就向后移动
        const int class_count = 1; // COCO类别数
        const int max_boxes_per_class = 1; // 每个类别最多100个框
        const float score_threshold = 0.4f; // 画框阈值

        // NMS BY CLASS 输出布局：每个类别先给 count，再给该类别最多100个框(每框5个float)
        for (int class_id = 0; class_id < class_count; ++class_id) {
			if (detection_offset >= detection_float_count) {
				break;
			}

			const int box_count = static_cast<int>(detections[detection_offset++]); // 该类别有效框数量
			for (int box_index = 0; box_index < max_boxes_per_class; ++box_index) {
				if (detection_offset + 4 >= detection_float_count) {
					break;
				}

				const float ymin = detections[detection_offset++]; // 框上边界
				const float xmin = detections[detection_offset++]; // 框左边界
				const float ymax = detections[detection_offset++]; // 框下边界
				const float xmax = detections[detection_offset++]; // 框右边界
				const float score = detections[detection_offset++]; // 置信度

				// 超过该类别有效数量，或分数太低，则跳过
				if (box_index >= box_count || score < score_threshold) {
					continue;
				}

				// 某些模型给0~1归一化坐标，某些模型给像素坐标；这里做兼容
				auto to_pixel_x = [&](float value) {
					return static_cast<int>(value <= 1.5f ? value * target_w : value);
				};
				auto to_pixel_y = [&](float value) {
					return static_cast<int>(value <= 1.5f ? value * target_h : value);
				};

				// 将坐标裁剪到画布范围，避免越界绘制
				const int x1 = std::max(0, std::min(to_pixel_x(xmin), target_w - 1));
				const int y1 = std::max(0, std::min(to_pixel_y(ymin), target_h - 1));
				const int x2 = std::max(0, std::min(to_pixel_x(xmax), target_w - 1));
				const int y2 = std::max(0, std::min(to_pixel_y(ymax), target_h - 1));

				// 非法框（右下在左上之前）直接丢弃
				if (x2 <= x1 || y2 <= y1) {
					continue;
				}

				// 计算边界框中心点
				const int center_x = (x1 + x2) / 2;
				const int center_y = (y1 + y2) / 2;

				// 将类别ID映射为COCO标签文本
				const std::string label = (class_id >= 0 && class_id < static_cast<int>(class_names.size()))
					? class_names[class_id]
					: (std::string("cls ") + std::to_string(class_id));

				// 绘制中心标出（红色圆点）
				cv::circle(display_frame, cv::Point(center_x, center_y), 2, cv::Scalar(255, 0, 0), -1);
				
				// 可选：在中心点上方显示标签和置信度
				cv::putText(display_frame,                                      // 图像
					label + " " + cv::format("%.2f", score),                    // 文本内容
					cv::Point(center_x - 20, std::max(0, center_y - 10)),       // 文本位置
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