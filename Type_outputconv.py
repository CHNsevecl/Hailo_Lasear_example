import onnx

# 加载模型
model = onnx.load('C:\\Develop\\Python\\pytorch\\蓝色激光识别模型\\runs\\detect\\runs\\train\\laser_det\\weights\\best.onnx')

print("=" * 50)
print("ONNX 模型输出节点信息")
print("=" * 50)

for i, output in enumerate(model.graph.output):
    print(f"输出 {i+1}:")
    print(f"  名称: {output.name}")
    print(f"  类型: {output.type}")
    print()

print("=" * 50)
print("所有输出节点名称列表:")
print("=" * 50)
output_names = [output.name for output in model.graph.output]
print(output_names)