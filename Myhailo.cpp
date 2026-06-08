#include "Myhailo.hpp"

#include <iostream>

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
	for (const auto &name : context.output_names) {
		const size_t output_size = context.infer_model->output(name)->get_frame_size();
		context.output_buffer.emplace_back(output_size);
		context.bindings.output(name)->set_buffer(hailort::MemoryView(context.output_buffer.back().data(), context.output_buffer.back().size()));
	}

	std::cout << "[Step 2 完成] 模型已配置，输入输出缓冲已绑定。" << std::endl;
	std::cout << "[Step 1 完成] 设备与模型检查通过。" << std::endl;

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
	
	auto run_status = hailo_context.configured_infer_model.run(hailo_context.bindings, std::chrono::milliseconds(1000));

	if (run_status != HAILO_SUCCESS) {
		std::cerr << "错误: 推理执行失败, status=" << run_status << std::endl;
		return std::nullopt;
	}

	const float *detections = reinterpret_cast<const float*>(hailo_context.output_buffer[0].data()); // 将输出缓冲按float数组解析
	const size_t detection_float_count = hailo_context.output_buffer[0].size() / sizeof(float); // 输出里一共有多少个float
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

			const int x1 = std::max(0, std::min(to_pixel_x(xmin), target_w - 1));
			const int y1 = std::max(0, std::min(to_pixel_y(ymin), target_h - 1));
			const int x2 = std::max(0, std::min(to_pixel_x(xmax), target_w - 1));
			const int y2 = std::max(0, std::min(to_pixel_y(ymax), target_h - 1));

			// 非法框（右下在左上之前）直接丢弃
			if (x2 <= x1 || y2 <= y1) {
				continue;
			}

			const std::string label = (class_id >= 0 && class_id < static_cast<int>(class_names.size()))
				? class_names[class_id]
				: (std::string("cls ") + std::to_string(class_id));

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

cv::Mat Stream_process(const cv::Mat& BGR_frame, int target_w, int target_h){
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

	return RGB_frame;
}