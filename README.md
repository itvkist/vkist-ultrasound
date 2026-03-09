# VKIST Ultrasound

## Introduction
**VKIST Ultrasound** is an application designed to support the diagnosis of **knee arthritis** using **knee ultrasound images**.  
The system processes ultrasound images and assists clinicians in identifying potential signs of arthritis, providing a supportive tool for medical analysis and research.

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

### Prerequisites

Before running the application, ensure the following are installed:

- **CUDA 12.4**
- **cuDNN 9.x** (compatible with CUDA 12.4)

If you are using a different CUDA version, please install the corresponding compatible version of PyTorch.

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