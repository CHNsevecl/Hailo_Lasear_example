import json
import shutil
import random
from pathlib import Path
from PIL import Image

# ============ 类别映射 ============
LABEL_TO_CLASS = {
    'Blue_Lasear': 0,
    'Canvas': 1,
}

def get_class_id(label):
    """根据 label 获取 class_id"""
    return LABEL_TO_CLASS.get(label, 0)


def fix_single_point_json(json_path, point_size=15):
    """修复只有 point 的 JSON，添加矩形框"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    needs_conversion = any(
        s.get('shape_type') == 'point' and s.get('group_id') is None
        for s in data.get('shapes', [])
    )
    
    if not needs_conversion:
        return False
    
    new_shapes = []
    group_id = 0
    
    existing_ids = [s.get('group_id') for s in data.get('shapes', []) if s.get('group_id') is not None]
    if existing_ids:
        group_id = max(existing_ids) + 1
    
    for shape in data.get('shapes', []):
        if shape.get('shape_type') == 'point' and shape.get('group_id') is None:
            x, y = shape['points'][0]
            half = point_size / 2
            img_w = data.get('imageWidth', 640)
            img_h = data.get('imageHeight', 640)
            
            x_min = max(0, x - half)
            y_min = max(0, y - half)
            x_max = min(img_w, x + half)
            y_max = min(img_h, y + half)
            
            bbox = {
                "label": shape.get('label', 'object'),
                "shape_type": "rectangle",
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": group_id
            }
            new_shapes.append(bbox)
            
            point = shape.copy()
            point['group_id'] = group_id
            new_shapes.append(point)
            
            group_id += 1
        else:
            new_shapes.append(shape)
    
    data['shapes'] = new_shapes
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"    ✓ 修复单点JSON: 添加了矩形框")
    return True


def json_to_yolo_txt(json_path, txt_path, img_width, img_height, point_size=15):
    """将 JSON 转换为 YOLO 格式 TXT，根据 label 分配 class_id"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'imageWidth' in data and 'imageHeight' in data:
        img_width = data['imageWidth']
        img_height = data['imageHeight']
    
    # 按 group_id 分组
    groups = {}
    standalone_shapes = []
    
    if 'shapes' in data:
        for shape in data['shapes']:
            gid = shape.get('group_id')
            if gid is not None:
                if gid not in groups:
                    groups[gid] = {'bbox': None, 'point': None, 'label': None}
                
                shape_type = shape.get('shape_type', '')
                points = shape.get('points', [])
                
                if shape_type == 'rectangle' and points:
                    groups[gid]['bbox'] = points
                    groups[gid]['label'] = shape.get('label', 'Blue_Lasear')
                elif shape_type == 'point' and points:
                    groups[gid]['point'] = points[0]
            else:
                standalone_shapes.append(shape)
    
    yolo_annotations = []
    
    # 处理有 group_id 的分组
    for gid, group in groups.items():
        bbox_points = group.get('bbox')
        label = group.get('label', 'Blue_Lasear')
        class_id = get_class_id(label)
        
        if bbox_points:
            if len(bbox_points) == 2:
                x1, y1 = bbox_points[0]
                x2, y2 = bbox_points[1]
            else:
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
            
            x1 = max(0, min(img_width, x1))
            y1 = max(0, min(img_height, y1))
            x2 = max(0, min(img_width, x2))
            y2 = max(0, min(img_height, y2))
            
            if abs(x2 - x1) < 1:
                x2 = x1 + 1
            if abs(y2 - y1) < 1:
                y2 = y1 + 1
            
            x_center = ((x1 + x2) / 2.0) / img_width
            y_center = ((y1 + y2) / 2.0) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 处理没有 group_id 的单个 shape（Canvas 单独矩形）
    for shape in standalone_shapes:
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])
        label = shape.get('label', 'Blue_Lasear')
        class_id = get_class_id(label)
        
        if not points:
            continue
        
        if shape_type == 'point':
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
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif shape_type == 'rectangle' and len(points) >= 2:
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
            
            x1 = max(0, min(img_width, x1))
            y1 = max(0, min(img_height, y1))
            x2 = max(0, min(img_width, x2))
            y2 = max(0, min(img_height, y2))
            
            if abs(x2 - x1) < 1:
                x2 = x1 + 1
            if abs(y2 - y1) < 1:
                y2 = y1 + 1
            
            x_center = ((x1 + x2) / 2.0) / img_width
            y_center = ((y1 + y2) / 2.0) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif shape_type == 'rotation' and len(points) >= 4:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min = max(0, min(xs))
            y_min = max(0, min(ys))
            x_max = min(img_width, max(xs))
            y_max = min(img_height, max(ys))
            
            if x_max - x_min < 1:
                x_max = x_min + 1
            if y_max - y_min < 1:
                y_max = y_min + 1
            
            x_center = ((x_min + x_max) / 2.0) / img_width
            y_center = ((y_min + y_max) / 2.0) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if yolo_annotations:
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    return len(yolo_annotations)


# ============ 配置 ============
TRAIN_DIR = Path("dataset/images/train")
VAL_DIR = Path("dataset/images/val")
LABELS_TRAIN = Path("dataset/labels/train")
LABELS_VAL = Path("dataset/labels/val")
VAL_SIZE = 30
POINT_SIZE = 15

# 创建目录
VAL_DIR.mkdir(parents=True, exist_ok=True)
LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
LABELS_VAL.mkdir(parents=True, exist_ok=True)

# ============ 第一步：修复所有单点 JSON ============
print("=" * 50)
print("第一步：修复单点 JSON（添加矩形框）")
print("=" * 50)

fixed_count = 0
for json_file in TRAIN_DIR.glob("*.json"):
    if fix_single_point_json(json_file, POINT_SIZE):
        fixed_count += 1
        print(f"  ✓ 修复: {json_file.name}")

print(f"\n共修复 {fixed_count} 个单点 JSON 文件\n")

# ============ 第二步：获取所有有 json 的图片 ============
images = [f for f in TRAIN_DIR.glob("*.jpg") if (TRAIN_DIR / f"{f.stem}.json").exists()]
print(f"有标注的图片: {len(images)}张")

# 随机选择一部分移动到 val
random.seed(42)
selected = random.sample(images, min(VAL_SIZE, len(images)))
print(f"移动到验证集: {len(selected)}张")

# ============ 第三步：移动图片并生成 txt ============
for img in selected:
    print(f"\n处理: {img.name}")
    
    shutil.move(str(img), str(VAL_DIR / img.name))
    json_file = TRAIN_DIR / f"{img.stem}.json"
    with Image.open(VAL_DIR / img.name) as pic:
        w, h = pic.size
    txt_file = LABELS_VAL / f"{img.stem}.txt"
    num_boxes = json_to_yolo_txt(json_file, txt_file, w, h, POINT_SIZE)
    print(f"  ✓ 生成了 {num_boxes} 个标注")

# ============ 第四步：为训练图片生成 txt ============
print(f"\n为剩余 {len(images)-len(selected)} 张训练图片生成 txt...")
for img in TRAIN_DIR.glob("*.jpg"):
    json_file = TRAIN_DIR / f"{img.stem}.json"
    if json_file.exists():
        txt_file = LABELS_TRAIN / f"{img.stem}.txt"
        if not txt_file.exists():
            with Image.open(img) as pic:
                w, h = pic.size
            num_boxes = json_to_yolo_txt(json_file, txt_file, w, h, POINT_SIZE)
            if num_boxes > 0:
                print(f"  ✓ {img.name} (转换了 {num_boxes} 个标注)")
            else:
                print(f"  ⚠ {img.name} (无有效标注)")

print("\n✅ 完成！")

# ============ 统计信息 ============
print("\n=== 统计信息 ===")
train_txt_count = len(list(LABELS_TRAIN.glob("*.txt")))
val_txt_count = len(list(LABELS_VAL.glob("*.txt")))
train_img_count = len(list(TRAIN_DIR.glob("*.jpg")))
val_img_count = len(list(VAL_DIR.glob("*.jpg")))
print(f"训练集图片: {train_img_count} 张, 标注文件: {train_txt_count} 个")
print(f"验证集图片: {val_img_count} 张, 标注文件: {val_txt_count} 个")