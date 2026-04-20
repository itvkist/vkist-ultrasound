# VKIST Ultrasound

## Introduction
**VKIST Ultrasound** is an application designed to support the diagnosis of **knee arthritis** using **knee ultrasound images**.  
The system processes ultrasound images and assists clinicians in identifying potential signs of arthritis, providing a supportive tool for medical analysis and research.

---

## Yêu cầu hệ thống

### 1. YÊU CẦU VỀ MÁY TÍNH
*   **Hệ điều hành:** Windows 10/11 (64-bit) hoặc Ubuntu 20.04/22.04.
*   **CPU:** Tối thiểu 4 nhân (khuyến nghị Intel Core i5 thế hệ 10 trở lên hoặc tương đương).
*   **RAM:** Tối thiểu 16GB.
*   **GPU:** NVIDIA GPU hỗ trợ CUDA (Kiến trúc Pascal trở lên, ví dụ: GTX 10-series, RTX series).
*   **VRAM:** Khuyến nghị 8GB trở lên để tối ưu tốc độ phân vùng (segmentation).

### 2. CÀI ĐẶT PHẦN MỀM HỖ TRỢ
*   **Quản lý môi trường:** [Anaconda3](https://www.anaconda.com/download) hoặc Miniconda.
*   **Đồ họa:** NVIDIA Driver tương thích với CUDA 12.4.
*   **CUDA Toolkit:** Phiên bản 12.4.
*   **Trình biên dịch C++:**
    *   **Windows:** [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    *   **Ubuntu:** Cài đặt qua lệnh `sudo apt install build-essential`.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/itvkist/vkist-ultrasound.git
cd vkist-ultrasound
```

### 2. Create Environment and Install Dependencies
```bash
conda create -n vkist-ultrasound python=3.10 -y
conda activate vkist-ultrasound
pip install -r requirements.txt
```

### Download Model Weights

The weights of the models can be found in the following link:

```
https://drive.google.com/drive/folders/1lBkplP-5uv6V2wR1CJ2COaGy1SnZxxJl
```

After downloading the link, unzip and copy the files into the `./models` folder

### Run the Application

Start the application with:

```bash
python app.py
```

The application will be available at:

```
http://localhost:8000
```

### Notes

- Make sure your GPU drivers are compatible with the CUDA version installed.
- If GPU support is not required, the PyTorch CPU version can also be used.