from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
# import easyocr  <- ĐÃ XÓA
import re
import cv2
import torchvision.models as models
import uvicorn
from pathlib import Path
import json
import zipfile
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, Any

# BẮT ĐẦU CODE MỚI: Tự động thêm đường dẫn code custom của UltraSam
import sys
import os

# Lấy đường dẫn tuyệt đối đến thư mục UltraSam (nằm cùng cấp với file app.py)
custom_code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'UltraSam'))

# Thêm đường dẫn này vào đầu danh sách tìm kiếm của Python
if custom_code_path not in sys.path:
    sys.path.insert(0, custom_code_path)
    print(f"✅ Successfully added custom code path to sys.path: {custom_code_path}")
# KẾT THÚC CODE MỚI

# BẮT ĐẦU CODE MỚI: Imports cho UltraSAM
try:
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules
    print("✅ MMDetection components loaded successfully.")
except ImportError:
    print("⚠️ MMDetection not found. UltraSAM model will be unavailable.")
    init_detector = None
    inference_detector = None
    register_all_modules = None
# KẾT THÚC CODE MỚI

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
TEMPLATES_FOLDER = 'templates'
EXPORTS_FOLDER = 'exports'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max

# Create directories
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, TEMPLATES_FOLDER, EXPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Model paths
EFFICIENTNET_CKPT_PATH = "models/efficientnet_b0_ultrasound_2_class.pth"
DEEPLABV3_CKPT_PATH = "models/deeplabv3.pth"
# THÊM MỚI: Đường dẫn đến model phân loại góc chụp (4 class)
POSITION_MODEL_CKPT_PATH = "models/best_model_angle_classification.pth"  # <<<--- Đường dẫn model 4 class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BẮT ĐẦU CODE MỚI: Cấu hình cho UltraSAM (6 class)
ULTRASAM_CONFIG_PATH = 'UltraSam/configs/UltraSAM/UltraSAM_full/downstream/segmentation/train_knee_resnet_6cls.py'
ULTRASAM_CKPT_PATH = 'models/converted_best_coco_segm_mAP_iter_5150.pth'
ULTRASAM_SCORE_THRESHOLD = 0.5 # Ngưỡng tin cậy
ULTRASAM_CLASSES = {
    0: 'effusion', 
    1: 'fat', 
    2: 'fat-pat', 
    3: 'femur', 
    4: 'tendon', 
    5: 'synovium'
}
# KẾT THÚC CODE MỚI


# Segmentation classes (Dùng chung cho cả DeepLabV3 và UltraSAM output)
SEGMENT_CLASSES = {
    # Key là class_id (bắt đầu từ 1 cho DeepLabV3, 0 cho UltraSAM)
    # Value là tên class chuẩn hóa
    1: 'effusion', 
    2: 'fat', 
    3: 'fat-pat', 
    4: 'femur', 
    5: 'tendon', 
    6: 'synovium'
}

# Position anatomy mapping
POSITION_ANATOMY = {
    'post-trans': {
        'angle': 'Góc chụp kheo chân đằng sau - ngang (Posterior Transverse)',
        'inflammation': 'Viêm nang Baker - Khu vực có thể có dịch khớp',
        'fluid_location': 'Nang Baker (Baker\'s cyst) - vùng dịch tập trung phía sau khớp gối'
    },
    'suprapat-long': { # Giữ key cũ làm fallback
        'angle': 'Góc chụp trường xương bánh chè - dọc (Suprapatellar Longitudinal)', 
        'inflammation': 'Viêm điểm bám gân tứ đầu - Khu vực có thể có dịch khớp',
        'fluid_location': 'Túi dịch trên xương bánh chè (Suprapatellar bursa) - vùng dịch tập trung phía trên bánh chè'
    },
    'sup-up-long': { # Thêm key chính xác từ model của bạn
        'angle': 'Góc chụp trường xương bánh chè - dọc (Suprapatellar Longitudinal)', 
        'inflammation': 'Viêm điểm bám gân tứ đầu - Khu vực có thể có dịch khớp',
        'fluid_location': 'Túi dịch trên xương bánh chè (Suprapatellar bursa) - vùng dịch tập trung phía trên bánh chè'
    },
    # <<<--- Bạn nên thêm mapping cho 'med-lat' và 'sup-trans-flex' vào đây
}

CLASSES = ['Viêm', 'Không viêm'] # Class cho model phân loại viêm

# THÊM MỚI: Tên class cho model phân loại góc chụp
POSITION_CLASS_NAMES = ['sup-up-long', 'med-lat', 'sup-trans-flex', 'post-trans']


# Pydantic models
class SaveAnalysisRequest(BaseModel):
    analysis_data: Dict[str, Any]
    doctor_notes: Optional[str] = ""
    original_path: Optional[str] = None
    segmented_path: Optional[str] = None

app = FastAPI(title="Integrated Medical Image Analysis API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/results", StaticFiles(directory=RESULTS_FOLDER), name="results")
app.mount("/exports", StaticFiles(directory=EXPORTS_FOLDER), name="exports")

print("Loading models...")

# -----------------------------------------------------------------
# KHỐI OCR READER ĐÃ BỊ XÓA
# -----------------------------------------------------------------

# Load EfficientNet Classification Model (Phân loại VIÊM 2 class)
inflammation_model = None
inflammation_transform = None
try:
    inflammation_model = models.efficientnet_b0(pretrained=False)
    inflammation_model.classifier[1] = torch.nn.Linear(inflammation_model.classifier[1].in_features, 2)
    inflammation_model = inflammation_model.to(device)
    
    if os.path.exists(EFFICIENTNET_CKPT_PATH):
        checkpoint = torch.load(EFFICIENTNET_CKPT_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            inflammation_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            inflammation_model.load_state_dict(checkpoint['model'])
        else:
            inflammation_model.load_state_dict(checkpoint)
        
        inflammation_model.eval()
        
        inflammation_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ EfficientNet Inflammation Classification model loaded")
    else:
        print(f"⚠️ EfficientNet (Inflammation) checkpoint not found: {EFFICIENTNET_CKPT_PATH}")
        inflammation_model = None
except Exception as e:
    print(f"❌ Error loading EfficientNet (Inflammation): {e}")

# THÊM MỚI: Tải mô hình phân loại GÓC CHỤP (Position Classification 4 class)
position_model = None
position_transform = None
try:
    print("Loading Position Classification model...")
    position_model = models.efficientnet_b0(pretrained=False)

    # Cấu trúc classifier bạn đã định nghĩa
    num_ftrs = position_model.classifier[1].in_features
    position_model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, len(POSITION_CLASS_NAMES))  # 4 classes
    )
    position_model = position_model.to(device)
    
    if os.path.exists(POSITION_MODEL_CKPT_PATH):
        checkpoint = torch.load(POSITION_MODEL_CKPT_PATH, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            position_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            position_model.load_state_dict(checkpoint['model'])
        else:
            position_model.load_state_dict(checkpoint)
        
        position_model.eval()
        
        position_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ EfficientNet Position Classification model loaded")
    else:
        print(f"⚠️ Position Model checkpoint not found: {POSITION_MODEL_CKPT_PATH}")
        position_model = None
except Exception as e:
    print(f"❌ Error loading Position Model: {e}")

# Load DeepLabV3 Segmentation Model
deeplabv3_model = None
deeplabv3_transform = None
try:
    deeplabv3_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    # SỬA ĐỔI: Tự động điều chỉnh theo 6 class
    deeplabv3_model.classifier[-1] = nn.Conv2d(256, len(SEGMENT_CLASSES) + 1, kernel_size=1)  # +1 for background
    
    if os.path.exists(DEEPLABV3_CKPT_PATH):
        state_dict = torch.load(DEEPLABV3_CKPT_PATH, map_location=device)
        deeplabv3_model.load_state_dict(state_dict)
        deeplabv3_model.to(device)
        deeplabv3_model.eval()
        
        deeplabv3_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        print("✅ DeepLabV3 Segmentation model loaded")
    else:
        print(f"⚠️ DeepLabV3 checkpoint not found: {DEEPLABV3_CKPT_PATH}")
        deeplabv3_model = None
except Exception as e:
    print(f"❌ Error loading DeepLabV3: {e}. (Lỗi này có thể xảy ra nếu checkpoint cũ là 4 class nhưng code đang mong đợi 6 class)")

# Load UltraSAM Segmentation Model
ultrasam_model = None
try:
    if register_all_modules and init_detector and os.path.exists(ULTRASAM_CONFIG_PATH) and os.path.exists(ULTRASAM_CKPT_PATH):
        register_all_modules()
        ultrasam_model = init_detector(ULTRASAM_CONFIG_PATH, ULTRASAM_CKPT_PATH, device=device)
        print("✅ UltraSAM Segmentation model loaded")
    else:
        if not register_all_modules:
            print("⚠️ UltraSAM loading failed: MMDetection library is not installed.")
        else:
            print(f"⚠️ UltraSAM config or checkpoint not found.")
except Exception as e:
    print(f"❌ Error loading UltraSAM: {e}")

# -----------------------------------------------------------------
# HÀM extract_position_from_image VÀ find_ultrasound_position ĐÃ BỊ XÓA
# -----------------------------------------------------------------

# THÊM MỚI: Hàm dự đoán góc chụp bằng model
@torch.no_grad()
def classify_position_efficientnet(image_pil):
    """Phân loại góc chụp sử dụng EfficientNet (4-class)"""
    if position_model is None or position_transform is None:
        print("Position classification model is not loaded.")
        return None, 0.0  # Trả về None nếu model không khả dụng
    
    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Áp dụng transform
        image_tensor = position_transform(image_pil).unsqueeze(0).to(device)
        
        # Dự đoán
        output = position_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class_idx].item()
        
        # Map index sang tên class (ví dụ: 'sup-up-long')
        class_label = POSITION_CLASS_NAMES[pred_class_idx]
        
        return class_label, confidence
        
    except Exception as e:
        print(f"EfficientNet Position Classification error: {e}")
        return None, 0.0 # Trả về None nếu có lỗi

def get_position_anatomy_info(position):
    """Get detailed anatomy information for position"""
    if not position:
        return None
    
    position_lower = position.lower().strip()
    
    if position_lower in POSITION_ANATOMY:
        return POSITION_ANATOMY[position_lower]
    
    for key_pos in POSITION_ANATOMY.keys():
        if key_pos in position_lower:
            return POSITION_ANATOMY[key_pos]
    
    return None

@torch.no_grad()
def classify_inflammation_efficientnet(image_pil):
    """Classify inflammation using EfficientNet"""
    if inflammation_model is None or inflammation_transform is None:
        return False, 0.5, "EfficientNet (Inflammation) model không khả dụng"
    
    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        image_tensor = inflammation_transform(image_pil).unsqueeze(0).to(device)
        
        output = inflammation_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        class_label = CLASSES[pred_class]
        is_inflammation = (pred_class == 0) # Giả sử 0 = Viêm, 1 = Không viêm
        
        return is_inflammation, confidence, class_label
        
    except Exception as e:
        print(f"EfficientNet Classification error: {e}")
        return False, 0.5, "Lỗi phân loại EfficientNet"

@torch.no_grad()
def segment_with_deeplabv3(image_pil):
    """Perform segmentation using DeepLabV3"""
    if deeplabv3_model is None or deeplabv3_transform is None:
        return None, None
    
    try:
        original_size = image_pil.size
        input_tensor = deeplabv3_transform(image_pil).unsqueeze(0).to(device)
        
        outputs = deeplabv3_model(input_tensor)['out']
        
        upsampled_logits = F.interpolate(outputs, size=original_size[::-1],
                                         mode='bilinear', align_corners=False)
        preds = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Create masks for each class
        masks = {}
        for class_id, class_name in SEGMENT_CLASSES.items():
            mask = (preds == class_id).astype(np.uint8)
            if np.sum(mask) > 0: # Chỉ thêm class nào được phát hiện
                masks[class_name] = mask
        
        return preds, masks
        
    except Exception as e:
        print(f"DeepLabV3 Segmentation error: {e}")
        return None, None

@torch.no_grad()
def segment_with_ultrasam(image_pil):
    """Perform segmentation using UltraSAM and format output to match DeepLabV3."""
    if ultrasam_model is None:
        return None, None

    try:
        img_np = np.array(image_pil.convert('RGB'))
        original_h, original_w, _ = img_np.shape

        # Run inference
        result = inference_detector(ultrasam_model, img_np)
        
        pred_instances = result.pred_instances
        
        # Khởi tạo dict mask rỗng
        masks = {class_name: np.zeros((original_h, original_w), dtype=np.uint8) for class_name in ULTRASAM_CLASSES.values()}
        
        # Khởi tạo preds map (0 là background)
        preds = np.zeros((original_h, original_w), dtype=np.uint8)

        if pred_instances and 'masks' in pred_instances and len(pred_instances.masks) > 0:
            for i in range(len(pred_instances.scores)):
                score = pred_instances.scores[i].item()
                label_id = pred_instances.labels[i].item()
                
                if score >= ULTRASAM_SCORE_THRESHOLD and label_id in ULTRASAM_CLASSES:
                    class_name = ULTRASAM_CLASSES[label_id]
                    mask_tensor = pred_instances.masks[i]
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    
                    # Gộp các mask cùng class
                    masks[class_name] = np.logical_or(masks[class_name], mask_np).astype(np.uint8)

            # Tạo 'preds' map (giống DeepLabV3)
            # Map tên class sang ID của SEGMENT_CLASSES (bắt đầu từ 1)
            name_to_id_map = {name: id for id, name in SEGMENT_CLASSES.items()}

            for class_name, mask in masks.items():
                if np.sum(mask) > 0 and class_name in name_to_id_map:
                    class_id = name_to_id_map[class_name]
                    preds[mask == 1] = class_id # Gán ID (1, 2, ... 6) vào preds
        
        # Lọc ra các mask rỗng
        final_masks = {name: mask for name, mask in masks.items() if np.sum(mask) > 0}

        return preds, final_masks

    except Exception as e:
        print(f"UltraSAM Segmentation error: {e}")
        return None, None

def analyze_inflammation_severity(masks, image_size):
    """
    SỬA ĐỔI: Phân tích mức độ viêm dựa trên 'effusion' (thay vì 'abnormal')
    """
    
    # SỬA ĐỔI: Tìm 'effusion'
    if not masks or 'effusion' not in masks:
        print("Analyze severity: 'effusion' class not found in masks.")
        return None
    
    # SỬA ĐỔI: Lấy mask 'effusion'
    abnormal_mask = masks['effusion']
    
    inflammation_pixels = int(np.sum(abnormal_mask))
    total_image_pixels = image_size[0] * image_size[1]
    ratio = (inflammation_pixels / total_image_pixels) * 100 if total_image_pixels > 0 else 0
    
    print(f"Inflammation analysis (effusion): {inflammation_pixels} pixels out of {total_image_pixels} total ({ratio:.2f}%)")
    
    if ratio > 8.0:
        level = 3
        severity = "Nặng"
        color = "#dc3545"
        description = "Vùng tràn dịch lan rộng, chiếm tỷ lệ lớn trong ảnh"
    elif ratio >= 3.0:
        level = 2
        severity = "Trung bình"
        color = "#fd7e14"
        description = "Vùng tràn dịch có kích thước trung bình"
    elif ratio >= 0.5:
        level = 1
        severity = "Nhẹ"
        color = "#ffc107"
        description = "Vùng tràn dịch nhỏ, tương đối hạn chế"
    else:
        level = 0
        severity = "Rất nhẹ"
        color = "#28a745"
        description = "Vùng tràn dịch rất nhỏ hoặc không đáng kể"
    
    return {
        'level': level,
        'severity': severity,
        'ratio': round(ratio, 2),
        'color': color,
        'description': description,
        'inflammation_pixels': inflammation_pixels,
        'total_image_pixels': total_image_pixels
    }


def analyze_segment_location(masks, position):
    """
    SỬA ĐỔI: Phân tích cấu trúc lân cận 'effusion' (thay vì 'abnormal')
    """
    
    # SỬA ĐỔI: Tìm 'effusion'
    if not masks or 'effusion' not in masks:
        print("Analyze location: 'effusion' class not found in masks.")
        return None
    
    # SỬA ĐỔI: Lấy mask 'effusion'
    abnormal_mask = masks['effusion']
    
    if np.sum(abnormal_mask) == 0:
        return None
    
    from scipy import ndimage
    
    expanded_abnormal = ndimage.binary_dilation(abnormal_mask, iterations=5)
    boundary_zone = expanded_abnormal & (~abnormal_mask)
    
    nearby_structures = {}
    boundary_pixels = np.sum(boundary_zone)
    
    if boundary_pixels == 0:
        return create_fallback_analysis(masks, position)
    
    for class_name, mask in masks.items():
        # SỬA ĐỔI: Bỏ qua 'effusion' và 'synovium' khi tìm lân cận
        if class_name != 'effusion' and class_name != 'synovium':
            nearby_pixels = np.sum(boundary_zone & mask)
            
            if nearby_pixels > 0:
                proximity_ratio = (nearby_pixels / boundary_pixels) * 100
                nearby_structures[class_name] = round(proximity_ratio, 1)
    
    description = create_proximity_description(nearby_structures)
    
    location_info = {
        'primary_location': 'proximity_analysis',
        'description': description,
        'nearby_structures': nearby_structures,
        'position_context': position
    }
    
    if position:
        if 'suprapat' in position.lower() or 'sup-up' in position.lower():
            location_info['context'] = 'Vùng suprapatellar - cần chú ý quan hệ với túi dịch và gân tứ đầu'
        elif 'post' in position.lower():
            location_info['context'] = 'Vùng posterior - cần đánh giá liên quan đến nang Baker'
    
    return location_info


def create_proximity_description(nearby_structures):
    """Tạo mô tả dựa trên cấu trúc gần kề (Hàm này vẫn đúng)"""
    if not nearby_structures:
        return "Không xác định được cấu trúc gần vùng viêm/tràn dịch"
    
    descriptions = []
    sorted_structures = sorted(nearby_structures.items(), key=lambda x: x[1], reverse=True)
    
    main_structure, main_ratio = sorted_structures[0]
    
    # Thêm class mới 'fat-pat'
    if main_structure == 'tendon':
        descriptions.append(f"Vùng viêm nằm giữa vùng gân ({main_ratio}% vùng xung quanh)")
    elif main_structure == 'fat' or main_structure == 'fat-pat':
        descriptions.append(f"Vùng viêm nằm trong/gần mô mỡ ({main_ratio}% vùng xung quanh)")
    elif main_structure == 'femur':
        descriptions.append(f"Vùng viêm nằm gần xương đùi ({main_ratio}% vùng xung quanh)")
    
    for structure, ratio in sorted_structures[1:]:
        if ratio > 20:
            if structure == 'tendon': descriptions.append(f"cũng tiếp cận vùng gân ({ratio}%)")
            elif structure == 'fat' or structure == 'fat-pat': descriptions.append(f"cũng liên quan đến mô mỡ ({ratio}%)")
            elif structure == 'femur': descriptions.append(f"cũng gần xương đùi ({ratio}%)")
            
    return "; ".join(descriptions)


def create_fallback_analysis(masks, position):
    """
    SỬA ĐỔI: Phân tích dự phòng cho 'effusion'
    """
    
    # SỬA ĐỔI: Lấy mask 'effusion'
    if 'effusion' not in masks:
        return None
    abnormal_mask = masks['effusion']

    total_pixels = np.sum(abnormal_mask)
    if total_pixels == 0:
        return None
        
    overlaps = {}
    for class_name, mask in masks.items():
        # SỬA ĐỔI: Bỏ qua 'effusion' và 'synovium'
        if class_name != 'effusion' and class_name != 'synovium':
            overlap = np.sum(abnormal_mask & mask)
            if overlap > 0:
                overlaps[class_name] = round((overlap / total_pixels) * 100, 1)
    
    if overlaps:
        main_structure = max(overlaps, key=overlaps.get)
        description = f"Vùng tràn dịch chồng lấp chủ yếu với {main_structure}"
    else:
        description = "Vùng tràn dịch được phân lập, không tiếp cận rõ ràng với các cấu trúc khác"
    
    return {
        'primary_location': 'fallback_analysis',
        'description': description,
        'overlaps': overlaps,
        'position_context': position
    }


def create_segmentation_overlay(image_pil, masks):
    """
    SỬA ĐỔI: Cập nhật color_map cho 6 class và SỬA LỖI OVERLAPPING
    """
    if masks is None or not masks: # Thêm kiểm tra 'not masks'
        return image_pil
    
    img_array = np.array(image_pil.convert('RGB')) # Chuyển sang RGB
    overlay = img_array.copy()
    
    color_map = {
        'effusion': [255, 0, 0],   # Tràn dịch - Đỏ
        'fat': [255, 255, 0],     # Mỡ - Vàng
        'fat-pat': [255, 165, 0], # Mỡ (patella) - Cam
        'femur': [0, 255, 0],     # Xương đùi - Xanh lá
        'tendon': [0, 0, 255],    # Gân - Xanh dương
        'synovium': [255, 0, 255]  # Màng hoạt dịch - Hồng/tím
    }

    # BẮT ĐẦU SỬA LỖI
    # 1. Định nghĩa thứ tự vẽ. Class ở cuối danh sách sẽ đè lên class ở đầu.
    # Chúng ta muốn 'effusion' và 'synovium' (các dấu hiệu viêm) ở trên cùng.
    draw_order = [
        'fat',
        'fat-pat',
        'femur',
        'tendon',
        'synovium',
        'effusion'
    ]

    for class_name in draw_order:
        # Chỉ vẽ nếu class này được phát hiện
        if class_name in masks and class_name in color_map:
            mask = masks[class_name]
            
            if np.sum(mask) > 0:
                color = color_map[class_name]
                for i in range(3):
                    # 2. Sửa logic blend:
                    # Luôn tính toán màu blend dựa trên 'img_array' (ảnh gốc)
                    # chứ không phải 'overlay' (ảnh đang được vẽ dở).
                    # Điều này đảm bảo màu cuối cùng là (Ảnh_gốc * 0.7 + Màu_class_cuối * 0.3)
                    
                    # Lấy màu đã blend (dựa trên ảnh gốc)
                    blended_color = (img_array[:, :, i] * 0.7 + color[i] * 0.3).astype(np.uint8)
                    
                    # Áp dụng màu này vào 'overlay' tại vị trí có mask
                    overlay[:, :, i] = np.where(mask > 0, 
                                                blended_color, 
                                                overlay[:, :, i])
    # KẾT THÚC SỬA LỖI
    
    return Image.fromarray(overlay)


def create_analysis_report(result_data, doctor_notes=""):
    """Create a comprehensive analysis report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": timestamp,
        "image_info": {
            "filename": result_data.get('filename'),
            "size": result_data.get('image_size')
        },
        "position_analysis": {
            "detected_position": result_data.get('position'),
            "position_confidence": result_data.get('position_confidence'), # Đã thêm
            "anatomy_info": result_data.get('anatomy_info')
        },
        "ai_analysis": {
            "classification": result_data.get('classification'),
            "segmentation_results": result_data.get('segmentation_analysis'),
            "location_analysis": result_data.get('location_analysis'),
            "inflammation_severity": result_data.get('inflammation_analysis')
        },
        "doctor_evaluation": {
            "notes": doctor_notes,
            "timestamp": timestamp
        }
    }
    
    return report

def save_analysis_package(result_data, original_image_path, segmented_image_path, doctor_notes=""):
    """Save complete analysis package as ZIP file"""
    try:
        package_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"analysis_{timestamp}_{package_id}"
        
        package_dir = os.path.join(EXPORTS_FOLDER, package_name)
        os.makedirs(package_dir, exist_ok=True)
        
        # Copy images
        original_dest = os.path.join(package_dir, "original_image.jpg")
        segmented_dest = os.path.join(package_dir, "segmented_image.jpg")
        
        if original_image_path and os.path.exists(original_image_path):
            import shutil
            shutil.copy2(original_image_path, original_dest)
        
        if segmented_image_path and os.path.exists(segmented_image_path):
            import shutil
            shutil.copy2(segmented_image_path, segmented_dest)
        
        # Create analysis report
        report = create_analysis_report(result_data, doctor_notes)
        report_path = os.path.join(package_dir, "analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Create human-readable summary
        summary_path = os.path.join(package_dir, "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO PHÂN TÍCH HÌNH ẢNH SIÊU ÂM\n")
            f.write(f"=====================================\n")
            f.write(f"Thời gian: {report['timestamp']}\n")
            f.write(f"Tên file: {report['image_info']['filename']}\n\n")
            
            f.write(f"VỊ TRÍ CHỤP:\n")
            f.write(f"- Vị trí phát hiện: {report['position_analysis']['detected_position'] or 'Không xác định'}\n")
            if report['position_analysis'].get('position_confidence'):
                f.write(f"- Độ tin cậy (vị trí): {report['position_analysis']['position_confidence']}%\n")
            if report['position_analysis']['anatomy_info']:
                f.write(f"- Góc chụp: {report['position_analysis']['anatomy_info'].get('angle', 'N/A')}\n")
            
            f.write(f"\nKẾT QUẢ PHÂN TÍCH TỰ ĐỘNG:\n")
            classification = report['ai_analysis']['classification']
            f.write(f"- Phân loại (viêm): {classification['label']}\n")
            f.write(f"- Độ tin cậy (viêm): {classification['confidence']}%\n")
            
            if report['ai_analysis'].get('inflammation_severity'):
                severity = report['ai_analysis']['inflammation_severity']
                f.write(f"- Mức độ (tràn dịch): {severity['severity']}\n")
                f.write(f"- Tỷ lệ vùng (tràn dịch): {severity['ratio']}% tổng diện tích ảnh\n")

            if report['ai_analysis']['location_analysis']:
                loc_analysis = report['ai_analysis']['location_analysis']
                f.write(f"- Mô tả vị trí (tràn dịch): {loc_analysis.get('description', 'N/A')}\n")

            f.write(f"\nNHẬN XÉT CỦA BÁC SĨ:\n")
            f.write(f"{doctor_notes or 'Chưa có nhận xét'}\n")
        
        # Create ZIP file
        zip_path = os.path.join(EXPORTS_FOLDER, f"{package_name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arcname)
        
        import shutil
        shutil.rmtree(package_dir)
        
        return f"/exports/{package_name}.zip"
        
    except Exception as e:
        print(f"Error creating analysis package: {e}")
        return None

# FastAPI Routes
@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    html_file = Path(TEMPLATES_FOLDER) / "index_ultrasam.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return JSONResponse({"error": "HTML template not found"})

@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    segment_model: str = Form("deeplabv3") # Giữ nguyên lựa chọn model
):
    """Main API endpoint for integrated image analysis"""
    try:
        # Validate file
        contents = await image.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(contents)
        
        image_pil = Image.open(filepath)
        
        # Step 1: Classify position using EfficientNet
        print("Classifying position...")
        position, position_confidence = classify_position_efficientnet(image_pil)
        anatomy_info = get_position_anatomy_info(position)
        
        # Step 2: Classification with EfficientNet (Viêm)
        is_inflammation, confidence, class_label = classify_inflammation_efficientnet(image_pil)
        
        # Step 3: Segmentation with selected model
        preds, masks = None, None
        segment_model_name = "None"

        if segment_model.lower() == 'ultrasam' and ultrasam_model:
            print("Performing segmentation with UltraSAM (6-class)...")
            preds, masks = segment_with_ultrasam(image_pil)
            segment_model_name = "UltraSAM"
        else:
            if deeplabv3_model:
                print("Performing segmentation with DeepLabV3 (6-class)...")
                preds, masks = segment_with_deeplabv3(image_pil)
                segment_model_name = "DeepLabV3"
        
        # Step 4: Analyze inflammation severity (đã được sửa để tìm 'effusion')
        inflammation_analysis = None
        if masks:
            print("Analyzing inflammation severity...")
            inflammation_analysis = analyze_inflammation_severity(masks, image_pil.size)
        
        # Step 5: Analyze segment location (đã được sửa để tìm 'effusion')
        location_analysis = None
        segmentation_performed = False
        
        if masks is not None and len(masks) > 0:
            segmentation_performed = True
            print("Analyzing segment location...")
            location_analysis = analyze_segment_location(masks, position)
            print("Creating segmentation overlay...")
            segmented_image = create_segmentation_overlay(image_pil, masks)
        else:
            print("No segments found or segmentation skipped.")
            segmented_image = image_pil # Trả về ảnh gốc nếu không có mask
        
        # Save result images
        segmented_filename = f"segmented_{filename}"
        segmented_path = os.path.join(RESULTS_FOLDER, segmented_filename)
        segmented_image.save(segmented_path)
        
        result = {
            'success': True,
            'filename': filename,
            'image_size': [image_pil.width, image_pil.height],
            
            'position': position,
            'position_confidence': round(position_confidence * 100, 1),
            'anatomy_info': anatomy_info,
            
            'model_used': f'efficientnet(position) + efficientnet(inflammation) + {segment_model_name}',
            
            'classification': {
                'has_inflammation': is_inflammation,
                'confidence': round(confidence * 100, 1),
                'label': class_label
            },
            'segmentation_performed': segmentation_performed,
            'segmentation_analysis': {
                'classes_detected': list(masks.keys()) if masks else [],
                'model': segment_model_name
            },
            'inflammation_analysis': inflammation_analysis,
            'location_analysis': location_analysis,
            'images': {
                'original': f"/uploads/{filename}",
                'segmented': f"/results/{segmented_filename}"
            }
        }
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-analysis")
async def save_analysis_endpoint(request_data: SaveAnalysisRequest):
    # (Hàm này giữ nguyên)
    try:
        analysis_data = request_data.analysis_data
        doctor_notes = request_data.doctor_notes or ""
        original_path = request_data.original_path
        segmented_path = request_data.segmented_path
        
        if not analysis_data:
            raise HTTPException(status_code=400, detail="Analysis data is required")
        
        original_full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(original_path)) if original_path else None
        segmented_full_path = os.path.join(RESULTS_FOLDER, os.path.basename(segmented_path)) if segmented_path else None
        
        download_path = save_analysis_package(analysis_data, original_full_path, segmented_full_path, doctor_notes)
        
        if download_path:
            return JSONResponse({
                'success': True,
                'message': 'Analysis saved successfully',
                'download_path': download_path
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to save analysis package")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in save_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        'status': 'healthy',
        'models': {
            'efficientnet_inflammation': inflammation_model is not None,
            'efficientnet_position': position_model is not None,
            'deeplabv3_segmentation': deeplabv3_model is not None,
            'ultrasam_segmentation': ultrasam_model is not None,
            # 'ocr_reader': Đã xóa
        },
        'device': str(device)
    })

if __name__ == '__main__':
    print("Integrated Medical Image Analysis FastAPI Server")
    print(f"http://localhost:8000")
    print(f"Health: http://localhost:8000/api/health") 
    print(f"Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)