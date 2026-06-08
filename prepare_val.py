import json
import shutil
import random
from pathlib import Path
from PIL import Image

def json_to_yolo_txt(json_path, txt_path, img_width, img_height, point_size=15):
    """将LabelMe格式的激光点JSON转换为YOLO格式TXT"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'imageWidth' in data and 'imageHeight' in data:
        img_width = data['imageWidth']
        img_height = data['imageHeight']
    
    yolo_annotations = []
    
    if 'shapes' in data:
        for shape in data['shapes']:
            if shape.get('shape_type') == 'point':
                points = shape.get('points', [])
                if points:
                    x, y = points[0]
                    
                    half_size = point_size / 2
                    x_min = max(0, x - half_size)
                    y_min = max(0, y - half_size)
                    x_max = min(img_width, x + half_size)
                    y_max = min(img_height, y + half_size)
                    
                    x_center = (x_min + x_max) / 2.0 / img_width
                    y_center = (y_min + y_max) / 2.0 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    return len(yolo_annotations)

# 配置
TRAIN_DIR = Path("dataset/images/train")
VAL_DIR = Path("dataset/images/val")
LABELS_TRAIN = Path("dataset/labels/train")
LABELS_VAL = Path("dataset/labels/val")
VAL_SIZE = 30  # 移动30张到验证集

# 创建目录
VAL_DIR.mkdir(parents=True, exist_ok=True)
LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
LABELS_VAL.mkdir(parents=True, exist_ok=True)

# 获取所有有json的图片
images = [f for f in TRAIN_DIR.glob("*.jpg") if (TRAIN_DIR / f"{f.stem}.json").exists()]
print(f"有标注的图片: {len(images)}张")

# 随机选择一部分移动到val
random.seed()
selected = random.sample(images, min(VAL_SIZE, len(images)))
print(f"移动到验证集: {len(selected)}张")

# 移动图片并生成txt
for img in selected:
    # 1. 移动图片
    shutil.move(str(img), str(VAL_DIR / img.name))
    
    # 2. 生成txt
    json_file = TRAIN_DIR / f"{img.stem}.json"
    with Image.open(VAL_DIR / img.name) as pic:
        w, h = pic.size
    txt_file = LABELS_VAL / f"{img.stem}.txt"
    json_to_yolo_txt(json_file, txt_file, w, h)
    
    print(f"  ✓ {img.name}")

# 为剩下的训练图片生成txt
print(f"\n为剩余{len(images)-len(selected)}张训练图片生成txt...")
for img in TRAIN_DIR.glob("*.jpg"):
    json_file = TRAIN_DIR / f"{img.stem}.json"
    if json_file.exists():
        txt_file = LABELS_TRAIN / f"{img.stem}.txt"
        if not txt_file.exists():  # 避免重复生成
            with Image.open(img) as pic:
                w, h = pic.size
            json_to_yolo_txt(json_file, txt_file, w, h)
            print(f"  ✓ {img.name}")

print("\n完成！")