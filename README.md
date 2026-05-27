# 🪸 SCoralDet: Thử Nghiệm Ablation Study Tăng Cường Ảnh Cho Huấn Luyện YOLOv12n Trực Quan

Hệ thống mã nguồn mở bóc tách module phục vụ nghiên cứu thực nghiệm (**Ablation Study**) các kỹ thuật xử lý ảnh nâng cao (White Balance, CLAHE, Gamma, Sharpen) trên nền tảng kiến trúc mô hình **YOLOv12 Nano** áp dụng cho bài toán phát hiện rạn san hô dưới biển.

Hệ thống được thiết kế theo tư duy đơn nhiệm (Modular Design) giúp **cố định dữ liệu kiểm thử tuyệt đối**, đảm bảo tính công bằng và chính xác khi so sánh các biểu đồ mAP giữa các lượt thí nghiệm khác nhau.

---

## 📁 Cấu Trúc Thư Mục Hệ Thống

```text
project_root/
│
├── raw_data/                  # [INPUT GỐC] Nơi chứa ảnh (.jpg/.png) và nhãn (.txt) thô ban đầu
│
├── configs/
│   ├── train_config.yaml      # Cấu hình siêu tham số huấn luyện (epochs, lr, optimizer...)
│   └── scoral.yaml            # Khai báo đường dẫn dữ liệu và danh sách class cho YOLOv12
│
├── enhancement/
│   └── generate_dataset.py    # [BƯỚC 2] Áp bộ lọc tăng cường ảnh linh hoạt từ Terminal
│
├── datasets/                  # Thư mục tự sinh (Không cần tạo thủ công)
│   ├── base_split/            # Lưu ảnh thô sạch đã chuẩn hóa hình học và chia 3 tập cố định
│   └── SCoralDet_enhanced/    # Lưu ảnh đích đã áp bộ lọc màu, nạp trực tiếp vào YOLO
│
├── results/                   # Thư mục tự sinh lưu ma trận mAP, đồ thị và báo cáo Excel
│
├── processing.py              # [BƯỚC 1] Chuẩn hóa hình học, ép tọa độ nhãn, chia tập cố định (Seed 42)
├── train.py                   # [BƯỚC 3] Khởi chạy tiến trình Fine-tune YOLOv12n (Ưu tiên GPU)
└── evaluate.py                # [BƯỚC 4] Kiểm thử sòng phẳng trên tập TEST & Xuất bản Excel


# 1. Sinh tập data màu áp dụng TẤT CẢ các bộ lọc (Kết hợp cả 4 cờ)
python enhancement/generate_dataset.py --wb --clahe --gamma --sharpen

# 2. Huấn luyện Fine-tune YOLOv12n trên tập dữ liệu tổng hợp này
python train.py

# 3. Chấm điểm sòng phẳng model trên tập TEST và ghi kết quả vào Excel
python evaluate.py --weights experiments/ablation_run/weights/best.pt