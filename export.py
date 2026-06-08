# YOLOv8 export to ONNX for Hailo compilation
import argparse
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Export YOLO model to ONNX')
parser.add_argument('--weights', type=str, default='runs/detect/runs/train/laser_det/weights/best.pt')
parser.add_argument('--out', type=str, default=None)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--format', type=str, default='onnx')
parser.add_argument('--opset', type=int, default=11)
parser.add_argument('--simplify', action='store_true')
parser.add_argument('--nms', action='store_true')
parser.add_argument('--dynamic', action='store_true')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

weights = args.weights
out = args.out or os.path.splitext(weights)[0] + '.' + args.format

print(f"Loading weights: {weights}")
model = YOLO(weights)
print(f"Export target: {out}")
export_path = model.export(
	format=args.format,
	imgsz=args.imgsz,
	opset=args.opset,
	simplify=args.simplify,
	nms=args.nms,
	dynamic=args.dynamic,
	device=args.device,
)

print(f"Export finished: {export_path}")
