import json
from pathlib import Path

def batch_convert_points_to_pose(annotation_dir, point_size=15):
    """批量转换目录下所有单点标注文件"""
    converted = 0
    for json_file in Path(annotation_dir).glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 检查是否需要转换（存在 group_id 为 null 的 point）
        needs_conversion = any(
            s.get('shape_type') == 'point' and s.get('group_id') is None
            for s in data.get('shapes', [])
        )
        
        if not needs_conversion:
            continue
        
        # 执行转换
        new_shapes = []
        group_id = 0
        
        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'point' and shape.get('group_id') is None:
                x, y = shape['points'][0]
                half = point_size / 2
                x_min = max(0, x - half)
                y_min = max(0, y - half)
                x_max = min(data['imageWidth'], x + half)
                y_max = min(data['imageHeight'], y + half)
                
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
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ 已转换: {json_file.name}")
        converted += 1
    
    print(f"\n完成！共转换 {converted} 个文件")

# 使用
batch_convert_points_to_pose("dataset/images/train", point_size=15)