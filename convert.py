import json
from pathlib import Path

JSON_DIR = Path("dataset/images/train")
LABEL_DIR = Path("dataset/labels/train")
LABEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAPPING = {"Blue_Lasear": 0}
BOX_SIZE = 0.03

for json_path in JSON_DIR.glob("*.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    if img_w is None or img_h is None:
        print(f"跳过 {json_path.name}: 无尺寸信息")
        continue

    # 生成检测框：用点标注中心补一个小框，适合 YOLOv8 detection 训练
    class_id = None
    cx = None
    cy = None
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in CLASS_MAPPING:
            continue
        class_id = CLASS_MAPPING[label]
        points = shape.get("points", [])
        if points:
            x, y = points[0]
            cx = x / img_w
            cy = y / img_h

    if cx is None or cy is None:
        (LABEL_DIR / json_path.with_suffix(".txt").name).write_text("")
        print(f"空文件: {json_path.name}")
        continue

    box_w = box_h = BOX_SIZE
    # 限制边界框不超出边界
    cx = max(box_w/2, min(1 - box_w/2, cx))
    cy = max(box_h/2, min(1 - box_h/2, cy))

    label_line = f"{class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"

    txt_path = LABEL_DIR / json_path.with_suffix(".txt").name
    txt_path.write_text(label_line)
    print(f"生成: {txt_path.name}")