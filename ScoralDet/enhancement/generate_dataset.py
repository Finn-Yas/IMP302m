import os
import argparse
import cv2
import numpy as np
import shutil
from pathlib import Path

def apply_white_balance(img):
    b, g, r = cv2.split(img)
    r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
    if r_avg == 0 or g_avg == 0 or b_avg == 0: return img
    k = (r_avg + g_avg + b_avg) / 3.0
    return cv2.merge([np.clip(b*(k/b_avg),0,255).astype(np.uint8), np.clip(g*(k/g_avg),0,255).astype(np.uint8), np.clip(r*(k/r_avg),0,255).astype(np.uint8)])

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_gamma(img, gamma=1.2):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def main():
    parser = argparse.ArgumentParser(description="Step 2: Ablation Data Enhancement Generator")
    parser.add_argument("--wb", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--gamma", action="store_true")
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--input", type=str, default="datasets/base_split")
    parser.add_argument("--output", type=str, default="datasets/SCoralDet_enhanced")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if output_dir.exists(): shutil.rmtree(output_dir)

    if not input_dir.exists():
        print("❌ Lỗi: Không tìm thấy dữ liệu nền từ processing.py! Hãy chạy nó trước.")
        return

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/current_ablation_status.txt", "w") as f:
        f.write(f"WB:{args.wb}\nCLAHE:{args.clahe}\nGAMMA:{args.gamma}\nSHARPEN:{args.sharpen}")

    for split in ["train", "val", "test"]:
        img_src_dir = input_dir / "images" / split
        lbl_src_dir = input_dir / "labels" / split
        
        img_dest_dir = output_dir / "images" / split
        lbl_dest_dir = output_dir / "labels" / split
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_src_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None: continue

            if args.wb:      img = apply_white_balance(img)
            if args.clahe:   img = apply_clahe(img)
            if args.gamma:   img = apply_gamma(img)
            if args.sharpen: img = apply_sharpen(img)

            cv2.imwrite(str(img_dest_dir / img_path.name), img)
            
            lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy(str(lbl_path), str(lbl_dest_dir / lbl_path.name))

    print(f"🎉 [XONG BƯỚC 2] Đã áp bộ lọc tăng cường lên cấu trúc dữ liệu nền.")

if __name__ == "__main__":
    main()