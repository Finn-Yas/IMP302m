import yaml
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Step 3: Fine-tune YOLOv12n with Default Augmentation")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--data", type=str, default="configs/scoral.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        train_config = yaml.safe_load(f)

    print("\n=================================================================")
    print("🔥 KHỞI CHẠY TIẾN TRÌNH HUẤN LUYỆN FINE-TUNE YOLOv12n")
    print("=================================================================")
    
    # Khởi tạo mô hình YOLOv12n
    model = YOLO("yolov12n.pt")
    
    # Tiến hành train (Mô hình tự động lưu best.pt dựa trên kết quả Val tốt nhất)
    model.train(data=args.data, project="experiments", name="ablation_run", **train_config)
    
    print("\n[HOÀN THÀNH BƯỚC 3] Tiến trình huấn luyện kết thúc. Trọng số tốt nhất đã được lưu!")

if __name__ == "__main__":
    main()