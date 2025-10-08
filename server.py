from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import time
import uuid
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage import transform as sk_transform
from segment_anything import sam_model_registry
import easyocr
import re
import cv2
from torchvision import transforms
import torchvision.models as models

# Import MedViT từ code của bạn
from MedViT import MedViT_tiny, MedViT_large
from ultrasound_dataset import UltrasoundTransform

app = Flask(__name__)
CORS(app, origins=['http://localhost:8080', 'http://localhost:8082', 'https://medsam.vkist-hub.com', 'https://pilot.vkist-hub.com'])  # Enable CORS for frontend

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model paths
MEDVIT_CKPT_PATH = "models/MedViT_large_knee2.pth"
EFFICIENTNET_CKPT_PATH = "models/efficientnet_b0_ultrasound_2_class.pth"  # Add your EfficientNet model path
SAM_MODEL_TYPE = "vit_b"
MEDSAM_CKPT_PATH = "models/medsam_model_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bounding box configurations
BOX_DICT = {
    (1024, 1024): (300, 300, 400, 400),
    (640, 480): (153, 113, 377, 214),
    (640, 452): (194, 128, 346, 185),
    (1920, 1080): (500, 300, 400, 400),
}

CLASSES = ['Viêm', 'Không viêm']

# Position anatomy mapping
POSITION_ANATOMY = {
    'post-trans': {
        'angle': 'Góc chụp kheo chân đằng sau - ngang (Posterior Transverse)',
        'inflammation': 'Viêm nang Baker - Khu vực có thể có dịch khớp',
        'fluid_location': 'Nang Baker (Baker\'s cyst) - vùng dịch tập trung phía sau khớp gối'
    },
    'suprapat-long': {
        'angle': 'Góc chụp trước xương bánh chè - dọc (Suprapatellar Longitudinal)', 
        'inflammation': 'Viêm điểm bám gân tứ đầu - Khu vực có thể có dịch khớp',
        'fluid_location': 'Túi dịch trên xương bánh chè (Suprapatellar bursa) - vùng dịch tập trung phía trên bánh chè'
    }
}

print("🔄 Loading models...")

# Initialize EasyOCR reader
ocr_reader = None
try:
    print("🔄 Loading OCR model...")
    ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("✅ EasyOCR model loaded")
except Exception as e:
    print(f"❌ Error loading OCR: {e}")

# Load MedViT Classification Model
medvit_model = None
medvit_transform = None
try:
    medvit_model = MedViT_large(num_classes=2).to(device)
    if os.path.exists(MEDVIT_CKPT_PATH):
        checkpoint = torch.load(MEDVIT_CKPT_PATH, map_location=device)
        medvit_model.load_state_dict(checkpoint['model'], strict=False)
        medvit_model.eval()
        medvit_transform = UltrasoundTransform(is_training=False)
        print("✅ MedViT Classification model loaded")
    else:
        print(f"❌ MedViT checkpoint not found: {MEDVIT_CKPT_PATH}")
        medvit_model = None
except Exception as e:
    print(f"❌ Error loading MedViT: {e}")

# Load EfficientNet Classification Model
efficientnet_model = None
efficientnet_transform = None
try:
    # Create EfficientNet model
    efficientnet_model = models.efficientnet_b0(pretrained=False)
    efficientnet_model.classifier[1] = torch.nn.Linear(efficientnet_model.classifier[1].in_features, 2)
    efficientnet_model = efficientnet_model.to(device)
    
    if os.path.exists(EFFICIENTNET_CKPT_PATH):
        checkpoint = torch.load(EFFICIENTNET_CKPT_PATH, map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            efficientnet_model.load_state_dict(checkpoint['model'])
        else:
            efficientnet_model.load_state_dict(checkpoint)
        
        efficientnet_model.eval()
        
        # EfficientNet preprocessing
        efficientnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ EfficientNet Classification model loaded")
    else:
        print(f"❌ EfficientNet checkpoint not found: {EFFICIENTNET_CKPT_PATH}")
        efficientnet_model = None
except Exception as e:
    print(f"❌ Error loading EfficientNet: {e}")

# Load MedSAM Segmentation Model
medsam_model = None
try:
    if os.path.exists(MEDSAM_CKPT_PATH):
        medsam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=MEDSAM_CKPT_PATH).to(device)
        medsam_model.eval()
        print("✅ MedSAM Segmentation model loaded")
    else:
        print(f"❌ MedSAM checkpoint not found: {MEDSAM_CKPT_PATH}")
except Exception as e:
    print(f"❌ Error loading MedSAM: {e}")

def extract_position_from_image(image_pil):
    """Extract ultrasound position using OCR with improved preprocessing"""
    if ocr_reader is None:
        return None
    
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Get image dimensions
        height, width = img_cv.shape[:2]
        
        # Focus on header area where position text usually appears (top 20% of image)
        header_height = int(height * 0.2)
        header_roi = img_cv[0:header_height, :]
        
        print(f"📐 Image size: {width}x{height}, Header ROI: {width}x{header_height}")
        
        # Preprocess header area for better OCR
        gray = cv2.cvtColor(header_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing approaches
        results_all = []
        
        # Method 1: Original image
        results_1 = ocr_reader.readtext(gray, paragraph=False, width_ths=0.8, height_ths=0.8)
        results_all.extend(results_1)
        
        # Method 2: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        results_2 = ocr_reader.readtext(enhanced, paragraph=False, width_ths=0.8, height_ths=0.8)
        results_all.extend(results_2)
        
        # Method 3: Thresholded
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results_3 = ocr_reader.readtext(thresh, paragraph=False, width_ths=0.8, height_ths=0.8)
        results_all.extend(results_3)
        
        # Method 4: Inverted (for dark text on light background)
        inverted = cv2.bitwise_not(thresh)
        results_4 = ocr_reader.readtext(inverted, paragraph=False, width_ths=0.8, height_ths=0.8)
        results_all.extend(results_4)
        
        print(f"📝 OCR found {len(results_all)} total detections")
        
        # Extract all detected text with confidence filtering
        detected_texts = []
        for (bbox, text, confidence) in results_all:
            if confidence > 0.4:  # Lower threshold for medical text
                clean_text = text.upper().strip()
                detected_texts.append(clean_text)
                print(f"OCR detected: '{text}' -> '{clean_text}' (confidence: {confidence:.3f})")
        
        # Remove duplicates while preserving order
        unique_texts = []
        seen = set()
        for text in detected_texts:
            if text not in seen:
                unique_texts.append(text)
                seen.add(text)
        
        print(f"📝 Unique texts: {unique_texts}")
        
        # Look for position patterns
        position = find_ultrasound_position(unique_texts)
        
        if position:
            print(f"🎯 Position detected: {position}")
            return position
        else:
            print("❌ No position pattern found in OCR results")
            return None
            
    except Exception as e:
        print(f"OCR error: {e}")
        return None

def find_ultrasound_position(texts):
    """Find ultrasound position from OCR text results with flexible keyword matching"""
    
    # Define keyword sets
    ANATOMICAL_LOCATIONS = {
        'SUPRAPAT', 'SUPRAPATELLAR', 'SUPRA', 
        'LAT', 'LATERAL',
        'MED', 'MEDIAL', 
        'POST', 'POSTERIOR',
        'ANT', 'ANTERIOR',
        'INFRAPATELLAR', 'INFRA'
    }
    
    ORIENTATIONS = {
        'LONG', 'LONGITUDINAL', 'LONGI',
        'TRANS', 'TRANSVERSE', 'TRANSV'
    }
    
    SIDES = {'L', 'R', 'LEFT', 'RIGHT'}
    
    # Normalize function
    def normalize_location(loc):
        loc_map = {
            'SUPRAPAT': 'SUPRAPAT', 'SUPRAPATELLAR': 'SUPRAPAT', 'SUPRA': 'SUPRAPAT',
            'LAT': 'LAT', 'LATERAL': 'LAT',
            'MED': 'MED', 'MEDIAL': 'MED',
            'POST': 'POST', 'POSTERIOR': 'POST',
            'ANT': 'ANT', 'ANTERIOR': 'ANT',
            'INFRAPATELLAR': 'INFRA', 'INFRA': 'INFRA'
        }
        return loc_map.get(loc, loc)
    
    def normalize_orientation(ori):
        ori_map = {
            'LONG': 'LONG', 'LONGITUDINAL': 'LONG', 'LONGI': 'LONG',
            'TRANS': 'TRANS', 'TRANSVERSE': 'TRANS', 'TRANSV': 'TRANS'
        }
        return ori_map.get(ori, ori)
    
    def normalize_side(side):
        side_map = {'LEFT': 'L', 'RIGHT': 'R', 'L': 'L', 'R': 'R'}
        return side_map.get(side, side)
    
    # Find keywords in texts
    found_locations = []
    found_orientations = []
    found_sides = []
    
    for text in texts:
        text_upper = text.upper().strip()
        
        # Check for anatomical locations
        if text_upper in ANATOMICAL_LOCATIONS:
            found_locations.append(normalize_location(text_upper))
            print(f"🎯 Found location: {text_upper} -> {normalize_location(text_upper)}")
        
        # Check for orientations
        if text_upper in ORIENTATIONS:
            found_orientations.append(normalize_orientation(text_upper))
            print(f"🎯 Found orientation: {text_upper} -> {normalize_orientation(text_upper)}")
        
        # Check for sides
        if text_upper in SIDES:
            found_sides.append(normalize_side(text_upper))
            print(f"🎯 Found side: {text_upper} -> {normalize_side(text_upper)}")
    
    # Also check for combined text patterns (fallback)
    combined_text = ' '.join(texts).upper()
    
    # Pattern matching as backup
    position_patterns = [
        r'([LR])\s*(LAT|LATERAL|SUPRAPAT|SUPRAPATELLAR|MED|MEDIAL|POST|POSTERIOR|ANT|ANTERIOR)\s*(LONG|LONGITUDINAL|TRANS|TRANSVERSE)',
        r'([LR])\s*(LAT|LATERAL|SUPRAPAT|SUPRAPATELLAR|MED|MEDIAL)',
        r'(LAT|LATERAL|SUPRAPAT|SUPRAPATELLAR|MED|MEDIAL)\s*(LONG|LONGITUDINAL|TRANS|TRANSVERSE)',
    ]
    
    for pattern in position_patterns:
        matches = re.findall(pattern, combined_text)
        if matches:
            match = matches[0]
            if len(match) >= 2:
                if len(match) == 3:  # Side + Location + Orientation
                    side, location, orientation = match
                    found_sides.append(normalize_side(side))
                    found_locations.append(normalize_location(location))
                    found_orientations.append(normalize_orientation(orientation))
                elif len(match) == 2:
                    # Could be Side+Location or Location+Orientation
                    if match[0] in ['L', 'R']:  # Side + Location
                        side, location = match
                        found_sides.append(normalize_side(side))
                        found_locations.append(normalize_location(location))
                    else:  # Location + Orientation
                        location, orientation = match
                        found_locations.append(normalize_location(location))
                        found_orientations.append(normalize_orientation(orientation))
                break
    
    # Build position string from found components
    position_parts = []
    
    # Add side (prefer L/R, but use any found side)
    if found_sides:
        position_parts.append(found_sides[0])
    
    # Add location
    if found_locations:
        position_parts.append(found_locations[0])
    
    # Add orientation
    if found_orientations:
        position_parts.append(found_orientations[0])
    
    if position_parts:
        position = ' '.join(position_parts)
        print(f"🎯 Constructed position: {position}")
        return position
    
    # Last resort: if we only found individual keywords, try to make sense
    if found_locations and found_orientations:
        position = f"{found_locations[0]} {found_orientations[0]}"
        print(f"🎯 Fallback position: {position}")
        return position
    
    if found_locations:
        position = found_locations[0]
        print(f"🎯 Location only: {position}")
        return position
    
    print("❌ No position could be constructed from OCR results")
    return None


def get_anatomy_details(position):
    """
    Get detailed anatomy information based on detected position
    Returns anatomy details if position matches known patterns, None otherwise
    """
    if not position:
        return None
    
    # Normalize position string for matching
    position_lower = position.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    # Check for post-trans pattern
    if 'post' in position_lower and 'trans' in position_lower:
        return POSITION_ANATOMY['post-trans']
    
    # Check for suprapat-long pattern  
    if ('suprapat' in position_lower or 'supra' in position_lower) and 'long' in position_lower:
        return POSITION_ANATOMY['suprapat-long']
    
    # Return None if no match found
    return None


@torch.no_grad()
def classify_inflammation_medvit(image_pil):
    """Classify inflammation using MedViT"""
    if medvit_model is None:
        return False, 0.5, "MedViT model không khả dụng"
    
    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        image_tensor = medvit_transform(image_pil).unsqueeze(0).to(device)
        
        start_time = time.perf_counter()
        output = medvit_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        end_time = time.perf_counter()
        
        infer_time = (end_time - start_time) * 1000
        class_label = CLASSES[pred_class]
        is_inflammation = (pred_class == 0)
        
        print(f"🔍 MedViT Classification: {class_label} (confidence: {confidence:.4f}, time: {infer_time:.2f}ms)")
        return is_inflammation, confidence, class_label
        
    except Exception as e:
        print(f"MedViT Classification error: {e}")
        return False, 0.5, "Lỗi phân loại MedViT"

@torch.no_grad()
def classify_inflammation_efficientnet(image_pil):
    """Classify inflammation using EfficientNet"""
    if efficientnet_model is None:
        return False, 0.5, "EfficientNet model không khả dụng"
    
    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        image_tensor = efficientnet_transform(image_pil).unsqueeze(0).to(device)
        
        start_time = time.perf_counter()
        output = efficientnet_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        end_time = time.perf_counter()
        
        infer_time = (end_time - start_time) * 1000
        # Invert the numerical output: 0->1, 1->0
        inverted_pred_class = 1 - pred_class
        
        # For EfficientNet, class 0 = inflammation, class 1 = no inflammation
        # But CLASSES array has: [0] = "Viêm", [1] = "Không viêm"
        # So we need to invert the logic for EfficientNet
        if inverted_pred_class == 0:
            class_label = "Viêm"
            is_inflammation = True
        else:
            class_label = "Không viêm"
            is_inflammation = False
        
        print(f"🔍 EfficientNet Classification: {class_label} (confidence: {confidence:.4f}, time: {infer_time:.2f}ms)")
        return is_inflammation, confidence, class_label
        
    except Exception as e:
        print(f"EfficientNet Classification error: {e}")
        return False, 0.5, "Lỗi phân loại EfficientNet"

def classify_inflammation(image_pil, model_name="medvit"):
    """Classify inflammation using selected model"""
    print(f"🔍 Using model: {model_name}")
    if model_name.lower() == "efficientnet":
        return classify_inflammation_efficientnet(image_pil)
    else:  # default to medvit
        return classify_inflammation_medvit(image_pil)

@torch.no_grad()
def get_embeddings(img_3c):
    """Get image embeddings for segmentation"""
    img_1024 = sk_transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)

    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    embedding = medsam_model.image_encoder(img_1024_tensor)
    return embedding

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    """Perform segmentation using MedSAM"""
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(height, width), mode="bilinear", align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def create_overlay_image(img_3c, sam_mask=None):
    """Create result image with mask overlay"""
    if sam_mask is None:
        return Image.fromarray(img_3c.astype(np.uint8))
    
    H, W = img_3c.shape[:2]
    
    # Create colored mask
    mask_colored = np.zeros((H, W, 3), dtype=np.uint8)
    mask_colored[sam_mask != 0] = [255, 255, 0]  # Yellow color
    
    # Blend original image with mask
    bg = Image.fromarray(img_3c.astype("uint8"), "RGB")
    mask = Image.fromarray(mask_colored.astype("uint8"), "RGB")
    result_img = Image.blend(bg, mask, 0.4)
    
    return result_img

def analyze_inflammation_severity(sam_mask, box_coords):
    """Analyze inflammation severity based on segmentation"""
    x, y, w, h = box_coords
    xmin, ymin, xmax, ymax = x, y, x + w, y + h
    
    # Get mask within bounding box
    mask_pixels = sam_mask[ymin:ymax, xmin:xmax]
    num_segmented_pixels = int(np.sum(mask_pixels))
    total_box_pixels = (ymax - ymin) * (xmax - xmin)
    ratio = num_segmented_pixels / total_box_pixels * 100 if total_box_pixels > 0 else 0
    
    # Determine severity
    if ratio > 25:
        level = 3
        severity = "Nặng"
        color = "#ff4444"
    elif ratio >= 15:
        level = 2
        severity = "Trung bình" 
        color = "#ff8800"
    elif ratio >= 5:
        level = 1
        severity = "Nhẹ"
        color = "#ffaa00"
    else:
        level = 0
        severity = "Rất nhẹ"
        color = "#ffdd00"
    
    return {
        'level': level,
        'severity': severity,
        'ratio': round(ratio, 1),
        'color': color,
        'pixels_count': num_segmented_pixels,
        'total_pixels': total_box_pixels
    }

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available classification models"""
    models = []
    
    if medvit_model is not None:
        models.append({
            'name': 'medvit',
            'display_name': 'MedViT Large',
            'description': 'Medical Vision Transformer - Specialized for medical imaging'
        })
    
    if efficientnet_model is not None:
        models.append({
            'name': 'efficientnet',
            'display_name': 'EfficientNet-B0',
            'description': 'EfficientNet - Efficient convolutional neural network'
        })
    
    return jsonify({
        'success': True,
        'models': models,
        'default': 'medvit' if medvit_model is not None else 'efficientnet'
    })

def generate_vietnamese_description(position, classification_result):
    """Generate Vietnamese description for the analysis"""
    descriptions = [
        "Hình ảnh siêu âm của khớp đầu gối",
        "Loại hình ảnh: Siêu âm",
        f"Vùng giải phẫu: {position if position else 'Khoang đầu gối trước'}",
        "Đặc điểm chính: Khe khớp, sụn khớp, dịch khớp",
        "Chất lượng hình ảnh: Độ phân giải và độ tương phản tốt"
    ]
    
    # Get detailed anatomy information if position matches known patterns
    anatomy_details = get_anatomy_details(position)
    if anatomy_details:
        descriptions.extend([
            f"Góc chụp: {anatomy_details['angle']}",
            f"Khu vực viêm có thể: {anatomy_details['inflammation']}",
            f"Vị trí dịch khớp: {anatomy_details['fluid_location']}"
        ])
    
    # Add classification-specific description
    if classification_result['has_inflammation']:
        descriptions.append(f"Phát hiện: {classification_result['label']} (độ tin cậy: {classification_result['confidence']}%)")
    else:
        descriptions.append(f"Phát hiện: {classification_result['label']} (độ tin cậy: {classification_result['confidence']}%)")
    
    return descriptions

def generate_vietnamese_prescription(severity_level, has_inflammation):
    """Generate Vietnamese prescription based on severity"""
    if not has_inflammation:
        return "Không phát hiện viêm khớp đầu gối. Duy trì sức khỏe tốt và tái khám định kỳ."
    
    prescriptions = {
        0: "Viêm khớp rất nhẹ. Theo dõi và tái khám định kỳ. Duy trì lối sống lành mạnh.",
        1: "Viêm khớp nhẹ. Điều trị được khuyến nghị: Bài tập vật lý trị liệu, thuốc chống viêm theo chỉ định của bác sĩ, quản lý cân nặng và tái khám sau 3 tháng.",
        2: "Viêm khớp trung bình. Điều trị được khuyến nghị: Bài tập vật lý trị liệu, thuốc chống viêm theo chỉ định của bác sĩ, quản lý cân nặng và tái khám sau 3 tháng. Tránh các hoạt động tác động mạnh và cân nhắc các bài tập tác động thấp như bơi lội hoặc đạp xe.",
        3: "Viêm khớp nặng. Cần điều trị tích cực: Thuốc chống viêm theo chỉ định của bác sĩ, vật lý trị liệu chuyên sâu, có thể cần can thiệp y tế. Tái khám thường xuyên và theo dõi chặt chẽ."
    }
    return prescriptions.get(severity_level, "Cần tư vấn bác sĩ để có phác đồ điều trị phù hợp.")

def convert_image_to_base64(image_path):
    """Convert image to base64 string"""
    import base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Main API endpoint for image analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Không có file hình ảnh được cung cấp'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        # Get mode and model parameters
        mode = request.form.get('mode', 'full')
        classification_model = request.form.get('classificationModel', request.form.get('model', 'medvit'))
        segmentation_model = request.form.get('segmentationModel', 'medsam')
        
        print(f"🤖 Mode: {mode}, Classification Model: {classification_model}")
        print(f"📋 All form data: {dict(request.form)}")
        
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess image
        image_pil = Image.open(filepath)
        img_np = np.array(image_pil)
        
        if img_np.ndim == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np
            
        H, W = img_3c.shape[:2]
        
        # Extract position using OCR
        print("🔤 Extracting position from image...")
        position = extract_position_from_image(image_pil)
        
        # Classification
        is_inflammation, confidence, class_label = classify_inflammation(image_pil, classification_model)
        classification_result = {
            'has_inflammation': is_inflammation,
            'confidence': round(confidence * 100, 1),
            'label': class_label
        }
        
        # Initialize result structure
        result = {
            'success': True,
            'message': 'Phân tích hoàn thành thành công',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'mode': mode,
            'model_used': classification_model,
            'processing_time': 0.0
        }
        
        # Mode-specific processing
        if mode == 'disease-classify':
            # Disease classification only
            description = generate_vietnamese_description(position, classification_result)
            prescription = generate_vietnamese_prescription(0, is_inflammation)
            
            result['data'] = {
                'inputImage': convert_image_to_base64(filepath),
                'description': description,
                'finalResult': 'Viêm Khớp Đầu Gối' if is_inflammation else 'Bình Thường',
                'prescription': prescription
            }
            
        elif mode == 'segmentation':
            # Segmentation only
            if medsam_model is not None:
                # Perform segmentation
                key = (W, H)
                if key not in BOX_DICT:
                    closest_key = min(BOX_DICT.keys(), 
                                    key=lambda k: abs(k[0] - W) + abs(k[1] - H))
                    box_coords = BOX_DICT[closest_key]
                else:
                    box_coords = BOX_DICT[key]
                
                x, y, w, h = box_coords
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
                
                embedding = get_embeddings(img_3c)
                box_np = np.array([[xmin, ymin, xmax, ymax]])
                box_1024 = box_np / np.array([W, H, W, H]) * 1024
                sam_mask = medsam_inference(medsam_model, embedding, box_1024, H, W)
                
                # Create result image with overlay
                result_image = create_overlay_image(img_3c, sam_mask)
                result_filename = f"result_{filename}"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                result_image.save(result_path)
                
                result['data'] = {
                    'inputImage': convert_image_to_base64(filepath),
                    'outputImages': [
                        {
                            'label': 'Kết Quả Phân Đoạn',
                            'url': convert_image_to_base64(result_path),
                            'type': 'segmentation'
                        }
                    ]
                }
            else:
                return jsonify({'error': 'Mô hình phân đoạn không khả dụng'}), 500
                
        else:  # full mode
            # Full analysis with both classification and segmentation
            description = generate_vietnamese_description(position, classification_result)
            
            # Determine severity level
            severity_level = 0
            if is_inflammation and medsam_model is not None:
                # Perform segmentation for severity analysis
                key = (W, H)
                if key not in BOX_DICT:
                    closest_key = min(BOX_DICT.keys(), 
                                    key=lambda k: abs(k[0] - W) + abs(k[1] - H))
                    box_coords = BOX_DICT[closest_key]
                else:
                    box_coords = BOX_DICT[key]
                
                x, y, w, h = box_coords
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
                
                embedding = get_embeddings(img_3c)
                box_np = np.array([[xmin, ymin, xmax, ymax]])
                box_1024 = box_np / np.array([W, H, W, H]) * 1024
                sam_mask = medsam_inference(medsam_model, embedding, box_1024, H, W)
                
                # Analyze inflammation severity
                inflammation_analysis = analyze_inflammation_severity(sam_mask, box_coords)
                severity_level = inflammation_analysis['level']
                
                # Create result image with overlay
                result_image = create_overlay_image(img_3c, sam_mask)
                result_filename = f"result_{filename}"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                result_image.save(result_path)
                
                result['data'] = {
                    'inputImage': convert_image_to_base64(filepath),
                    'outputImages': [
                        {
                            'label': 'Kết Quả Phân Đoạn',
                            'url': convert_image_to_base64(result_path),
                            'type': 'segmentation'
                        }
                    ],
                    'description': description,
                    'finalResult': f'Viêm Khớp Đầu Gối Cấp Độ {severity_level}' if severity_level > 0 else 'Bình Thường',
                    'prescription': generate_vietnamese_prescription(severity_level, is_inflammation)
                }
            else:
                result['data'] = {
                    'inputImage': convert_image_to_base64(filepath),
                    'description': description,
                    'finalResult': 'Viêm Khớp Đầu Gối' if is_inflammation else 'Bình Thường',
                    'prescription': generate_vietnamese_prescription(severity_level, is_inflammation)
                }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'error': f'Lỗi phân tích: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'medvit_classification': medvit_model is not None,
            'efficientnet_classification': efficientnet_model is not None,
            'medsam_segmentation': medsam_model is not None,
            'ocr_reader': ocr_reader is not None
        },
        'device': str(device)
    })

if __name__ == '__main__':
    print("🚀 Medical Image Analysis API Server")
    print(f"📍 http://localhost:8000")
    print(f"🔧 Health: http://localhost:8000/api/health")
    print(f"🤖 Available models: http://localhost:8000/api/models")
    app.run(host='0.0.0.0', port=8000, debug=True)