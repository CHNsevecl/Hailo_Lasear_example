#include "Myhailo.hpp"
#include <iostream>

#include <fstream>


// 在 run_status == HAILO_SUCCESS 之后插入这段代码
void SaveOutputToFile(const std::vector<uint8_t>& buffer) {
    const float* data = reinterpret_cast<const float*>(buffer.data());
    size_t float_count = buffer.size() / sizeof(float); // 应该等于 1002

    std::ofstream outFile("hailo_debug_output.txt");
    if (!outFile.is_open()) {
        std::cerr << "无法创建调试文件！" << std::endl;
        return;
    }

    outFile << "--- Hailo NMS Raw Data Dump ---" << std::endl;
    outFile << "Total floats: " << float_count << std::endl;

    for (int class_id = 0; class_id < 2; ++class_id) {
        int class_base = class_id * 501;
        float count = data[class_base];

        outFile << "\n========================================" << std::endl;
        outFile << "CLASS ID: " << class_id << " | DETECTED COUNT: " << count << std::endl;
        outFile << "========================================" << std::endl;

        // 记录该类别的所有 100 个框的空间
        for (int b = 0; b < 100; ++b) {
            int box_base = class_base + 1 + (b * 5);
            float ymin = data[box_base + 0];
            float xmin = data[box_base + 1];
            float ymax = data[box_base + 2];
            float xmax = data[box_base + 3];
            float conf = data[box_base + 4];

            // 无论有没有目标，我们都打印前 5 个，剩下的只打印有分数的
            if (b < 5 || conf > 0.01f) {
                outFile << "Box " << std::setw(2) << b << ": "
                        << "Score: " << std::fixed << std::setprecision(4) << conf << " | "
                        << "Rect: [" << ymin << ", " << xmin << ", " << ymax << ", " << xmax << "]" 
                        << (b >= count ? " (Invalid/Padding)" : " (Valid)")
                        << std::endl;
            }
        }
    }

    outFile.close();
    // std::cout << "调试数据已保存到 hailo_debug_output.txt" << std::endl;
}

std::optional<HailoContext> Hailo_init(const std::string& hef_path) {
	HailoContext context;

	std::cout << "[Step 1] 初始化 Hailo 设备并读取 HEF..." << std::endl;

	auto vdevice_exp = hailort::VDevice::create();
	if (!vdevice_exp) {
		std::cerr << "错误: 无法创建设备, status=" << vdevice_exp.status() << std::endl;
		return std::nullopt;
	}
	context.vdevice = vdevice_exp.release();

	auto infer_model_exp = context.vdevice->create_infer_model(hef_path);
	if (!infer_model_exp) {
		std::cerr << "错误: 无法加载 HEF: " << hef_path << ", status=" << infer_model_exp.status() << std::endl;
		return std::nullopt;
	}
	context.infer_model = infer_model_exp.release();

	context.input_names = context.infer_model->get_input_names();
	context.output_names = context.infer_model->get_output_names();

	std::cout << "input_names size = " << context.input_names.size() << std::endl;
	for (const auto &n : context.input_names) std::cout << "  " << n << std::endl;
	std::cout << "output_names size = " << context.output_names.size() << std::endl;
	for (const auto &n : context.output_names) std::cout << "  " << n << std::endl;

	if (context.input_names.empty() || context.output_names.empty()) {
		std::cerr << "错误: 模型输入或输出为空" << std::endl;
		return std::nullopt;
	}

	if (context.output_names.size() != 1) {
		std::cerr << "错误: 当前模型有 " << context.output_names.size() << " 个输出节点，"
				  << "此程序仅支持单输出模型，请检查模型文件" << std::endl;
		return std::nullopt;
	}
	std::cout << "模型输出节点数量检查通过 (仅支持单输出): " << context.output_names.size() << std::endl;

	for (const auto &name : context.input_names) {
		const size_t input_size = context.infer_model->input(name)->get_frame_size();
		std::cout << "  - " << name << ": " << input_size << " bytes" << std::endl;
	}

	std::cout << "输出节点数量: " << context.output_names.size() << std::endl;
	for (const auto &name : context.output_names) {
		const size_t output_size = context.infer_model->output(name)->get_frame_size();
		std::cout << "  - " << name << ": " << output_size << " bytes" << std::endl;
	}

	auto configured_infer_model_exp = context.infer_model->configure();
	if (!configured_infer_model_exp) {
		std::cerr << "错误: 无法配置模型, status=" << configured_infer_model_exp.status() << std::endl;
		return std::nullopt;
	}
	context.configured_infer_model = configured_infer_model_exp.release();

	auto bindings_exp = context.configured_infer_model.create_bindings();
	if (!bindings_exp) {
		std::cerr << "错误: 无法创建 bindings, status=" << bindings_exp.status() << std::endl;
		return std::nullopt;
	}
	context.bindings = bindings_exp.release();

	context.input_buffer.reserve(context.input_names.size());
	for (const auto &name : context.input_names) {
		const size_t input_size = context.infer_model->input(name)->get_frame_size();
		context.input_buffer.emplace_back(input_size);
		context.bindings.input(name)->set_buffer(hailort::MemoryView(context.input_buffer.back().data(), context.input_buffer.back().size()));
	}

	context.output_buffer.reserve(context.output_names.size());
	std::cout << context.output_names.size() << " output buffers will be allocated and bound." << std::endl;
	for (const auto &name : context.output_names) {
		const size_t output_size = context.infer_model->output(name)->get_frame_size();
		context.output_buffer.emplace_back(output_size);
		context.bindings.output(name)->set_buffer(hailort::MemoryView(context.output_buffer.back().data(), context.output_buffer.back().size()));
	}

	std::cout << "[Step 1 完成] 设备与模型检查通过。" << std::endl;
	std::cout << "[Step 2 完成] 模型已配置，输入输出缓冲已绑定。" << std::endl;
	

	return context;
}


std::optional<std::vector<Detection>> ParseDetections(HailoContext& hailo_context, int target_w, int target_h, cv::Mat& RGB_frame,const std::vector<std::string>& class_names) {
	std::vector<Detection> results;

	if (!RGB_frame.isContinuous()) {//copyTo、Rect等操作可能导致图像数据不连续，克隆一份确保数据连续
		RGB_frame = RGB_frame.clone();
	}
	//流数据处理完毕

	//流数据放入输入内存

	// input_buffers[0].size()                          // 模型输入缓冲区大小（字节）恰好是yolov8s,只有一个输入
	// resized_rgb.total()                              // 图像像素总数（宽×高）
	// resized_rgb.elemSize()                           // 每个像素的字节数
	// resized_rgb.total() * resized_rgb.elemSize()     // 图像数据总大小（字节）
	if (hailo_context.input_buffer[0].size() != RGB_frame.total() * RGB_frame.elemSize()) {
		std::cerr << "错误: 摄像头预处理后的输入大小与模型输入不一致" << std::endl;
		return std::nullopt;
	}

	std::memcpy(&hailo_context.input_buffer[0][0], RGB_frame.data, hailo_context.input_buffer[0].size());
	// 这两个是等价的,都是获取第一个元素的指针地址
	// hailo_context.input_buffer[0].data()
	// &hailo_context.input_buffer[0][0]
	
	std::memset(hailo_context.output_buffer[0].data(), 0xCC, 4008);
	cv::Mat check_mat(640, 640, CV_8UC3, hailo_context.input_buffer[0].data());
	cv::imwrite("hailo_input_actual.jpg", check_mat);
	// std::cout << "已保存硬件输入快照，请检查颜色和形状是否正常" << std::endl;
	auto run_status = hailo_context.configured_infer_model.run(hailo_context.bindings, std::chrono::milliseconds(1000));
	SaveOutputToFile(hailo_context.output_buffer[0]); // 调试：将原始输出保存到文件

	if (run_status != HAILO_SUCCESS) {
		std::cerr << "错误: 推理执行失败, status=" << run_status << std::endl;
		return std::nullopt;
	}

	const float *detections = reinterpret_cast<const float*>(hailo_context.output_buffer[0].data()); // 将输出缓冲按float数组解析
	const size_t detection_float_count = hailo_context.output_buffer[0].size() / sizeof(float); // 输出里一共有多少个float
	

	// NMS BY CLASS 输出布局：每个类别先给 count，再给该类别最多100个框(每框5个float)
	for (size_t class_id = 0; class_id < class_names.size(); ++class_id) {
		int class_base = class_id * 501; // 绝对基准点
    	int box_count = static_cast<int>(detections[class_base]); // 该类别有效框数量


		for (int box_index = 0; box_index < box_count; ++box_index) { // 只遍历有数据的框
			int box_base = class_base + 1 + (box_index * 5); // 每个框的绝对起始点

			const float ymin  = detections[box_base + 0];
			const float xmin  = detections[box_base + 1];
			const float ymax  = detections[box_base + 2];
			const float xmax  = detections[box_base + 3];
			const float score = detections[box_base + 4];

			if (score < score_threshold) continue;

			// 某些模型给0~1归一化坐标，某些模型给像素坐标；这里做兼容
			auto to_pixel_x = [&](float value) {
				return static_cast<int>(value <= 1.5f ? value * target_w : value);
			};
			auto to_pixel_y = [&](float value) {
				return static_cast<int>(value <= 1.5f ? value * target_h : value);
			};

			const int x1 = std::max(0, std::min(to_pixel_x(xmin), target_w - 1));
			const int y1 = std::max(0, std::min(to_pixel_y(ymin), target_h - 1));
			const int x2 = std::max(0, std::min(to_pixel_x(xmax), target_w - 1));
			const int y2 = std::max(0, std::min(to_pixel_y(ymax), target_h - 1));

			//非法框（右下在左上之前）直接丢弃
			if (x2 <= x1 || y2 <= y1) {
				continue;
			}

			const std::string label = class_names[class_id];

			Detection det;
			det.label = label;
			det.upper = cv::Point(x1, y1);
			det.lower = cv::Point(x2, y2);
			det.score = score;
			results.push_back(det);
		}
	}
	return std::make_optional(results);
}

cv::Mat Stream_process(const cv::Mat& BGR_frame, int target_w, int target_h) {
    // 计算缩放比例
    const float scale = std::min(static_cast<float>(target_w) / BGR_frame.cols, static_cast<float>(target_h) / BGR_frame.rows);

    const int new_w = static_cast<int>(BGR_frame.cols * scale);
    const int new_h = static_cast<int>(BGR_frame.rows * scale);
    const int x_offset = (target_w - new_w) / 2;
    const int y_offset = (target_h - new_h) / 2;

    // --- 修改点：不要在原图上 resize，创建一个中间变量 ---
    cv::Mat resized_frame;
    cv::resize(BGR_frame, resized_frame, cv::Size(new_w, new_h));

    // 创建黑色背景
    cv::Mat letterbox_bgr(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    
    // 将缩放后的图拷贝到黑色背景中心
    resized_frame.copyTo(letterbox_bgr(cv::Rect(x_offset, y_offset, new_w, new_h)));

    // 颜色空间转换 BGR -> RGB
    cv::Mat RGB_frame;
    cv::cvtColor(letterbox_bgr, RGB_frame, cv::COLOR_BGR2RGB);

    return RGB_frame;
}