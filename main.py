# train.py
from ultralytics import YOLO
import torch

def main():
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    model = YOLO('yolov8s.pt')
    
    # 开始训练
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        lr0=0.01,
        patience=30,
        save=True,
        save_period=10,
        project='runs/train',
        name='laser_det',
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    print("训练完成！")
    print(f"最佳模型保存在: runs/train/laser_det/weights/best.pt")

if __name__ == '__main__':
    main()