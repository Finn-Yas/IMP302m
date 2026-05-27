import os
import argparse
import random
import shutil
import cv2
from pathlib import Path

def validate_and_normalize_yolo_labels(lbl_path):
    valid_lines = []
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return None
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            try:
                cls_id = int(parts[0])
                x = max(0.0, min(1.0, float(parts[1])))
                y = max(0.0, min(1.0, float(parts[2])))
                w = max(0.0, min(1.0, float(parts[3])))
                h = max(0.0, min(1.0, float(parts[4])))
                valid_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            except ValueError: continue
    return valid_lines if len(valid_lines) > 0 else None

def main():
    parser = argparse.ArgumentParser(description="Step 1: Core Data Standardization & Fixed Splitting")
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa ảnh + nhãn gốc")
    parser.add_argument("--img_size", type=int, default=640, help="Kích thước hình học chuẩn hóa")
    parser.add_argument("--output", type=str, default="datasets/base_split")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists(): shutil.rmtree(output_dir)

    data_path = Path(args.data_dir)
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        img_files.extend(list(data_path.glob(ext)))

    valid_pairs = []
    for img_p in img_files:
        lbl_p = img_p.with_suffix('.txt')
        cleaned_labels = validate_and_normalize_yolo_labels(lbl_p)
        if cleaned_labels:
            valid_pairs.append((img_p, cleaned_labels))

    random.seed(42)
    random.shuffle(valid_pairs)

    idx_train = int(len(valid_pairs) * 0.70)
    idx_val = int(len(valid_pairs) * 0.85)
    splits = {"train": valid_pairs[:idx_train], "val": valid_pairs[idx_train:idx_val], "test": valid_pairs[idx_val:]}

    for split_name, pairs in splits.items():
        img_dest = output_dir / "images" / split_name
        lbl_dest = output_dir / "labels" / split_name
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for img_path, label_lines in pairs:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(img_dest / img_path.name), img)
            
            with open(lbl_dest / f"{img_path.stem}.txt", "w") as f_lbl:
                f_lbl.writelines(label_lines)

    print(f"🎉 [XONG BƯỚC 1] Dữ liệu nền cố định đã lưu tại: {args.output}")
    print(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

if __name__ == "__main__":
    main()