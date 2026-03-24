from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import uvicorn
from pathlib import Path
import cv2
import io
import base64

# Import custom models
import sys
sys.path.append('.')
from arch.efficientfeedback import EfficientFeedbackNetwork
from arch.unet3plus_att import UNet3Plus_Attention

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
TEMPLATES_FOLDER = 'templates'

# for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, TEMPLATES_FOLDER]:
#     os.makedirs(folder, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
ANGLE_CLASSES = ['med-lat', 'post-trans', 'sup-trans-flex', 'sup-up-long']
SEGMENT_CLASSES_SUPRAPAT = {0: "background", 1: "effusion", 2: "fat", 3: "fat-pat", 4: "femur", 5: "synovium", 6: "tendon"}
SEGMENT_CLASSES_POST = {0: "background", 1: "fat", 2: "tendon", 3: "muscle", 4: "femur", 5: "artery", 6: "baker's cyst"}

# Color map for Suprapat
COLOR_MAP_SUP = {
    'background': {'color': [0, 0, 0], 'name': 'Nền'},
    'effusion': {'color': [255, 0, 0], 'name': 'Dịch khớp'},
    'fat': {'color': [255, 255, 0], 'name': 'Mỡ'},
    'fat-pat': {'color': [0, 255, 255], 'name': 'Mỡ Hoffa'},
    'femur': {'color': [0, 255, 0], 'name': 'Xương đùi'},
    'synovium': {'color': [255, 0, 255], 'name': 'Màng hoạt dịch'},
    'tendon': {'color': [0, 0, 255], 'name': 'Gân'}
}

# Color map for Post-trans
COLOR_MAP_POST = {
    'background': {'color': [0, 0, 0], 'name': 'Nền'},
    "baker's cyst": {'color': [255, 0, 0], 'name': "Baker's cyst"},
    'fat': {'color': [255, 255, 0], 'name': 'Mỡ'},
    'muscle': {'color': [0, 255, 255], 'name': 'Cơ bắp'},
    'femur': {'color': [0, 255, 0], 'name': 'Xương đùi'},
    'synovium': {'color': [255, 0, 255], 'name': 'Màng hoạt dịch'},
    'tendon': {'color': [0, 0, 255], 'name': 'Gân'}
}

# Measurement configuration
DEFAULT_MEASURE_IDS = [1, 5]
PIXEL_TO_MM = 45.0 / 655.0

app = FastAPI(title="Medical Image Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
# app.mount("/results", StaticFiles(directory=RESULTS_FOLDER), name="results")

# ============ MODEL LOADING ============

def load_angle_model(model_name):
    print(f"📄 Loading angle model: {model_name}")
    
    if model_name == "convnext":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
        checkpoint = torch.load(f"models/best_convnext_tiny.pth", map_location=device, weights_only=False)
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    elif model_name == "densenet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 4)
        checkpoint = torch.load(f"models/best_densenet.pth", map_location=device, weights_only=False)
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        checkpoint = torch.load(f"models/best_resnet50.pth", map_location=device, weights_only=False)
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
        checkpoint = torch.load(f"models/best_efficientnet_b2.pth", map_location=device, weights_only=False)
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    elif model_name == "swin":
        model = models.swin_v2_s(weights=None)
        model.head = nn.Linear(model.head.in_features, 4)
        checkpoint = torch.load(f"models/best_swin_v2_s.pth", map_location=device, weights_only=False)
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unknown angle model: {model_name}")
    
    print(f"✅ Loaded: {model_name}")
    return model.to(device).eval()

def load_inflammation_model():
    print("📄 Loading inflammation model")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("models/efficientnet_b0_ultrasound_2_class.pth", map_location=device, weights_only=False))
    print("✅ Loaded inflammation model")
    return model.to(device).eval()

def load_segmentation_model_sup(model_name):
    print(f"📄 Loading segmentation model SUP: {model_name}")
    
    if model_name == "deeplabv3":
        model = models.segmentation.deeplabv3_resnet50(weights=None)
        in_ch = model.classifier[-1].in_channels
        model.classifier = nn.Sequential(
            model.classifier[0],
            nn.Dropout(0.3),
            nn.Conv2d(in_ch, 7, kernel_size=1)
        )
        model.load_state_dict(torch.load("models/best_model_Deeplav3.pth", map_location=device, weights_only=False), strict=False)
    elif model_name == "unet_resnet101":
        try:
            from segmentation_models_pytorch import Unet
            model = Unet(encoder_name="resnet101", encoder_weights=None, classes=7)
            model.load_state_dict(torch.load("models/unet_resnet101.pth", map_location=device, weights_only=False))
        except ImportError:
            raise ValueError("segmentation_models_pytorch not installed")
    elif model_name == "efficientfeedback":
        model = EfficientFeedbackNetwork(in_channels=3, num_class=7)
        model.load_state_dict(torch.load("models/efficientfeedback.pth", map_location=device, weights_only=False))
    elif model_name == "unet3plus":
        model = UNet3Plus_Attention(in_channels=3, num_classes=7)
        model.load_state_dict(torch.load("models/unet3plus_att.pth", map_location=device, weights_only=False))
    else:
        raise ValueError(f"Unknown segmentation model: {model_name}")
    
    print(f"✅ Loaded SUP: {model_name}")
    return model.to(device).eval()

def load_segmentation_model_post(model_name):
    print(f"📄 Loading segmentation model POST: {model_name}")
    
    if model_name == "deeplabv3_resnet101":
        model = models.segmentation.deeplabv3_resnet101(weights=None)
        in_ch = model.classifier[-1].in_channels
        model.classifier = nn.Sequential(
            model.classifier[0],
            nn.Dropout(0.3),
            nn.Conv2d(in_ch, 7, kernel_size=1)
        )
        model.load_state_dict(torch.load("models/best_model_deeplabv3_resnet101_seed_16.pth", map_location=device, weights_only=False), strict=False)
    else:
        raise ValueError(f"Unknown post segmentation model: {model_name}")
    
    print(f"✅ Loaded POST: {model_name}")
    return model.to(device).eval()

# Transforms
angle_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inflammation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

segmentation_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# ============ PREDICTION ============

@torch.no_grad()
def predict_angle(model, image_pil):
    img_tensor = angle_transform(image_pil).unsqueeze(0).to(device)
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    return ANGLE_CLASSES[pred_class], round(confidence * 100, 2)

@torch.no_grad()
def predict_inflammation(model, image_pil):
    img_tensor = inflammation_transform(image_pil).unsqueeze(0).to(device)
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    is_inflammation = (pred_class == 1)
    return is_inflammation, round(confidence * 100, 2)

@torch.no_grad()
def segment_image(model, image_pil, model_type, angle_type):
    original_size = image_pil.size
    img_tensor = segmentation_transform(image_pil).unsqueeze(0).to(device)
    
    if model_type.startswith("deeplabv3"):
        outputs = model(img_tensor)['out']
    else:
        outputs = model(img_tensor)
    
    upsampled = F.interpolate(outputs, size=original_size[::-1], mode='bilinear', align_corners=False)
    preds = upsampled.argmax(dim=1)[0].cpu().numpy()
    
    if angle_type == 'sup' and model_type in ["unet3plus", "efficientfeedback"]:
        remap = {0: 0, 1: 2, 2: 6, 3: 1, 4: 4, 5: 5, 6: 3}
        preds = np.vectorize(remap.get)(preds)
    
    class_map = SEGMENT_CLASSES_SUPRAPAT if angle_type == 'sup' else SEGMENT_CLASSES_POST
    
    masks = {}
    for class_id, class_name in class_map.items():
        mask = (preds == class_id).astype(np.uint8)
        masks[class_name] = mask
    
    return preds, masks

def find_max_continuous_segment(col_array):
    padded = np.concatenate(([0], col_array, [0]))
    diffs = np.diff(padded)
    
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    if len(starts) == 0:
        return 0, -1, -1
    
    lengths = ends - starts
    max_idx = np.argmax(lengths)
    max_len = lengths[max_idx]
    
    return max_len, starts[max_idx], ends[max_idx]

def measure_thickness_new(masks, image_size, measure_ids=None):
    if measure_ids is None:
        measure_ids = DEFAULT_MEASURE_IDS
    
    width, height = image_size
    
    mask_all_labels = np.zeros((height, width), dtype=np.uint8)
    mask_measure = np.zeros((height, width), dtype=np.uint8)
    
    has_any_label = False

    if 'fat-pat' in masks:
        class_map = SEGMENT_CLASSES_SUPRAPAT
    else:
        class_map = SEGMENT_CLASSES_POST

    for class_id, class_name in class_map.items():
        if class_name not in masks:
            continue
        mask = masks[class_name]
        if np.sum(mask) > 0:
            has_any_label = True
            mask_all_labels = np.logical_or(mask_all_labels, mask).astype(np.uint8)
            
            if class_id in measure_ids:
                mask_measure = np.logical_or(mask_measure, mask).astype(np.uint8)
    
    if not has_any_label or np.sum(mask_measure) == 0:
        return None
    
    points = cv2.findNonZero(mask_all_labels)
    if points is None:
        return None
    
    x_all, y_all, w_all, h_all = cv2.boundingRect(points)
    
    roi_start = x_all + (w_all // 3)
    roi_end = x_all + (2 * w_all // 3)
    
    roi_strip = mask_measure[:, roi_start:roi_end]
    
    global_max_len_px = 0
    best_x_rel = 0
    best_y_start = 0
    best_y_end = 0
    
    for x in range(roi_strip.shape[1]):
        col = roi_strip[:, x]
        if not np.any(col):
            continue
        
        length, y_s, y_e = find_max_continuous_segment(col)
        
        if length > global_max_len_px:
            global_max_len_px = length
            best_x_rel = x
            best_y_start = y_s
            best_y_end = y_e
    
    if global_max_len_px == 0:
        return None
    
    thickness_mm = global_max_len_px * PIXEL_TO_MM
    real_x = roi_start + best_x_rel
    
    print(f"📏 Measurement: {thickness_mm:.2f}mm ({global_max_len_px}px) at x={real_x}")
    
    return {
        'thickness_px': int(global_max_len_px),
        'thickness_mm': float(round(thickness_mm, 2)),
        'x': int(real_x),
        'y_start': int(best_y_start),
        'y_end': int(best_y_end),
        'roi_start': int(roi_start),
        'roi_end': int(roi_end),
        'bbox': {'x': int(x_all), 'y': int(y_all), 'w': int(w_all), 'h': int(h_all)}
    }

def analyze_inflammation_severity(masks, image_size):
    if not masks:
        return None
    
    width, height = image_size
    total_pixels = width * height
    
    effusion_mask = masks.get('effusion', np.zeros((height, width), dtype=np.uint8))
    effusion_pixels = int(np.sum(effusion_mask))
    effusion_ratio = (effusion_pixels / total_pixels) * 100
    
    effusion_thickness = 0
    if effusion_pixels > 0:
        rows_with_effusion = np.any(effusion_mask > 0, axis=1)
        if np.any(rows_with_effusion):
            effusion_thickness = int(np.sum(rows_with_effusion))
    
    synovium_mask = masks.get('synovium', np.zeros((height, width), dtype=np.uint8))
    synovium_pixels = int(np.sum(synovium_mask))
    synovium_ratio = (synovium_pixels / total_pixels) * 100
    
    effusion_score = min(effusion_thickness / height * 100, 100)
    synovium_score = synovium_ratio
    combined_score = (effusion_score * 0.6 + synovium_score * 0.4)
    
    if combined_score > 15:
        level, severity, color = 3, "Nặng", "#dc3545"
        description = f"Dịch khớp dày ({effusion_thickness}px), màng hoạt dịch tăng sinh rõ"
    elif combined_score >= 8:
        level, severity, color = 2, "Trung bình", "#fd7e14"
        description = f"Dịch khớp trung bình ({effusion_thickness}px), màng hoạt dịch tăng sinh vừa"
    elif combined_score >= 3:
        level, severity, color = 1, "Nhẹ", "#ffc107"
        description = f"Dịch khớp mỏng ({effusion_thickness}px), màng hoạt dịch tăng sinh nhẹ"
    else:
        level, severity, color = 0, "Rất nhẹ", "#28a745"
        description = "Lượng dịch và màng hoạt dịch trong giới hạn bình thường"
    
    return {
        'level': int(level),
        'severity': severity,
        'color': color,
        'description': description,
        'effusion': {'pixels': effusion_pixels, 'ratio': float(round(effusion_ratio, 2)), 'thickness': effusion_thickness},
        'synovium': {'pixels': synovium_pixels, 'ratio': float(round(synovium_ratio, 2))},
        'combined_score': float(round(combined_score, 2))
    }

def create_segmentation_overlay(image_pil, masks, measurement=None, angle_type='sup'):
    if masks is None:
        return image_pil
    
    color_map = COLOR_MAP_SUP if angle_type == 'sup' else COLOR_MAP_POST
    
    img_array = np.array(image_pil)
    overlay = img_array.copy()
    
    for class_name, mask in masks.items():
        if class_name in color_map and np.sum(mask) > 0:
            color = color_map[class_name]['color']
            for i in range(3):
                overlay[:, :, i] = np.where(mask > 0, 
                                          (overlay[:, :, i] * 0.6 + color[i] * 0.4).astype(np.uint8), 
                                          overlay[:, :, i])
    
    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)
    
    for class_name in ['effusion', 'synovium']:
        mask = masks.get(class_name)
        if mask is not None and np.sum(mask) > 0:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                points = contour.reshape(-1, 2).tolist()
                if len(points) > 2:
                    points = [(int(p[0]), int(p[1])) for p in points]
                    draw.line(points + [points[0]], fill=(255, 255, 255), width=3)
    
    if measurement and angle_type == 'sup':
        x = measurement['x']
        y_start = measurement['y_start']
        y_end = measurement['y_end']
        thickness_mm = measurement['thickness_mm']
        roi_start = measurement['roi_start']
        roi_end = measurement['roi_end']
        bbox = measurement['bbox']
        
        draw.rectangle(
            [bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']],
            outline=(255, 200, 0),
            width=2
        )
        
        h = image_pil.size[1]
        draw.line([(roi_start, 0), (roi_start, h)], fill=(0, 255, 255), width=2)
        draw.line([(roi_end, 0), (roi_end, h)], fill=(0, 255, 255), width=2)
        
        draw.line([(x, y_start), (x, y_end)], fill=(255, 0, 0), width=4)
        
        radius = 4
        draw.ellipse([x-radius, y_start-radius, x+radius, y_start+radius], 
                     fill=(0, 255, 0), outline=(255, 255, 255), width=2)
        draw.ellipse([x-radius, y_end-radius, x+radius, y_end+radius], 
                     fill=(0, 255, 0), outline=(255, 255, 255), width=2)
        
        text = f"{thickness_mm:.2f} mm"
        
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except:
            text_w, text_h = 100, 20
        
        text_x = x + 8
        text_y = y_start - text_h - 8
        
        draw.rectangle(
            [text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2],
            fill=(0, 0, 0)
        )
        draw.text((text_x, text_y), text, fill=(255, 0, 0))
    
    return overlay_pil


# Mount thư mục static (CSS, JS)
app.mount("/css", StaticFiles(directory="templates/css"), name="css")
app.mount("/js", StaticFiles(directory="templates/js"), name="js")

@app.get("/")
async def read_index():
    html_file = Path(TEMPLATES_FOLDER) / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return JSONResponse({"error": "Template not found"})

@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    angle_model: str = Query("convnext"),
    inflammation_model: str = Query("efficientnet_b0"),
    segment_model_sup: str = Query("deeplabv3"),
    segment_model_post: str = Query("deeplabv3_resnet101")
):
    try:
        print(f"\n{'='*60}")
        print(f"📊 NEW REQUEST")
        print(f"Models: angle={angle_model}, inflam={inflammation_model}")
        print(f"        seg_sup={segment_model_sup}, seg_post={segment_model_post}")
        print(f"{'='*60}")
        
        contents = await image.read()
        # filename = f"{uuid.uuid4().hex}_{image.filename}"
        # filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # with open(filepath, "wb") as f:
        #     f.write(contents)
        
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result = {
            'success': True,
            'filename': image.filename,
            'images': {},
            'models_used': {
                'angle': angle_model,
                'inflammation': inflammation_model,
                'segmentation_sup': segment_model_sup,
                'segmentation_post': segment_model_post
            }
        }
        
        angle_clf = load_angle_model(angle_model)
        angle, angle_conf = predict_angle(angle_clf, image_pil)
        result['angle'] = {'class': angle, 'confidence': angle_conf}
        print(f"✅ Angle: {angle} ({angle_conf}%)")
        
        if 'post-trans' in angle.lower():
            print(f"🔍 POST-TRANS pipeline")
            
            inflam_model = load_inflammation_model()
            has_inflammation, inflam_conf = predict_inflammation(inflam_model, image_pil)
            result['inflammation'] = {'detected': has_inflammation, 'confidence': inflam_conf}
            print(f"✅ Inflammation: {has_inflammation} ({inflam_conf}%)")
            
            seg_model = load_segmentation_model_post(segment_model_post)
            preds, masks = segment_image(seg_model, image_pil, segment_model_post, 'post')
            
            if masks:
                segmented_img = create_segmentation_overlay(image_pil, masks, None, 'post')
                # Convert to base64
                buffered = io.BytesIO()
                segmented_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                result['images']['segmented'] = f"data:image/png;base64,{img_str}"
                
                detected_classes = [k for k, v in masks.items() if np.sum(v) > 0]
                color_legend = []
                for class_name in detected_classes:
                    if class_name in COLOR_MAP_POST:
                        color_legend.append({
                            'name': COLOR_MAP_POST[class_name]['name'],
                            'color': COLOR_MAP_POST[class_name]['color'],
                            'key': class_name
                        })
                
                result['segmentation'] = {
                    'performed': True,
                    'classes_detected': detected_classes,
                    'color_legend': color_legend,
                    'angle_type': 'post'
                }
                
                print(f"✅ Segmentation POST completed")
        
        elif 'sup-up-long' in angle.lower():
            print(f"🔍 SUPRAPAT pipeline")
            
            inflam_model = load_inflammation_model()
            has_inflammation, inflam_conf = predict_inflammation(inflam_model, image_pil)
            result['inflammation'] = {'detected': has_inflammation, 'confidence': inflam_conf}
            print(f"✅ Inflammation: {has_inflammation} ({inflam_conf}%)")
            
            seg_model = load_segmentation_model_sup(segment_model_sup)
            preds, masks = segment_image(seg_model, image_pil, segment_model_sup, 'sup')
            
            if masks:
                measurement = measure_thickness_new(masks, image_pil.size)
                
                segmented_img = create_segmentation_overlay(image_pil, masks, measurement, 'sup')
                # Convert to base64
                buffered = io.BytesIO()
                segmented_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                result['images']['segmented'] = f"data:image/png;base64,{img_str}"
                
                if measurement:
                    result['measurement'] = {
                        'thickness_mm': measurement['thickness_mm'],
                        'thickness_px': measurement['thickness_px'],
                        'location_x': measurement['x'],
                        'y_start': measurement['y_start'],
                        'y_end': measurement['y_end']
                    }
                    print(f"✅ Measurement: {measurement['thickness_mm']:.2f}mm at x={measurement['x']}")
                
                detected_classes = [k for k, v in masks.items() if np.sum(v) > 0]
                color_legend = []
                for class_name in detected_classes:
                    if class_name in COLOR_MAP_SUP:
                        color_legend.append({
                            'name': COLOR_MAP_SUP[class_name]['name'],
                            'color': COLOR_MAP_SUP[class_name]['color'],
                            'key': class_name
                        })
                
                result['segmentation'] = {
                    'performed': True,
                    'classes_detected': detected_classes,
                    'color_legend': color_legend,
                    'angle_type': 'sup'
                }
                
                severity = analyze_inflammation_severity(masks, image_pil.size)
                if severity:
                    result['severity'] = severity
                    print(f"✅ Severity: {severity['severity']}")
                
                print(f"✅ Segmentation SUP completed")
        
        else:
            print(f"ℹ️ Other angle - only angle classification")
        
        print(f"{'='*60}\n")
        return JSONResponse(result)
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return JSONResponse({'status': 'healthy'})

class SaveDataRequest(BaseModel):
    patient_info: dict
    analysis_result: dict
    images: dict

@app.post("/api/save")
async def save_patient_data(data: SaveDataRequest):
    try:
        p = data.patient_info
        res = data.analysis_result
        imgs = data.images
        
        # 1. Tạo thư mục theo mã bệnh nhân
        patient_id = p.get('id', 'unknown').strip()
        patient_name = p.get('name', 'no_name').strip()
        
        # Clean folder name
        folder_name = f"{patient_id}_{patient_name}".replace(" ", "_")
        target_dir = os.path.join("patients", folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 2. Lưu info.txt
        info_path = os.path.join(target_dir, "info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"--- THÔNG TIN BỆNH NHÂN ---\n")
            f.write(f"Mã BN: {patient_id}\n")
            f.write(f"Họ tên: {patient_name}\n")
            f.write(f"Giới tính: {p.get('gender')}\n")
            f.write(f"Tuổi: {p.get('age')}\n")
            f.write(f"Chẩn đoán BS: {p.get('diagnosis')}\n\n")
            
            f.write(f"--- KẾT QUẢ PHÂN TÍCH AI ---\n")
            f.write(f"Góc chụp: {res.get('angle', {}).get('class')} ({res.get('angle', {}).get('confidence')}%)\n")
            
            if 'inflammation' in res:
                infl = res['inflammation']
                f.write(f"Viêm nhiễm: {'Có' if infl['detected'] else 'Không'} ({infl['confidence']}%)\n")
            
            if 'measurement' in res:
                m = res['measurement']
                f.write(f"Độ dày màng: {m['thickness_mm']} mm ({m['thickness_px']} px)\n")
                f.write(f"Vị trí x: {m['location_x']}\n")
            
            if 'severity' in res:
                s = res['severity']
                f.write(f"Mức độ: {s['severity']}\n")
                f.write(f"Mô tả: {s['description']}\n")

        # 3. Lưu ảnh
        def save_base64_img(b64_str, filename):
            if not b64_str: return
            # Remove header if present
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            
            img_data = base64.b64decode(b64_str)
            with open(os.path.join(target_dir, filename), "wb") as f:
                f.write(img_data)

        save_base64_img(imgs.get('original'), "original.png")
        save_base64_img(imgs.get('segmented'), "segmented.png")
        
        print(f"✅ Data saved for patient: {patient_id}")
        return {"success": True, "folder": target_dir}
        
    except Exception as e:
        print(f"❌ Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    print("🏥 Medical Image Analysis Server")
    print(f"🌐 http://127.0.0.1:8000")
    print(f"🖥️ Device: {device}")
    uvicorn.run(app, host="127.0.0.1", port=8000)