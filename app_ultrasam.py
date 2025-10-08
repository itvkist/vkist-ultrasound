from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import time
import uuid
import numpy as np
from PIL import Image
import torch
import mmcv
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
import easyocr
import re
import cv2
from torchvision import transforms

# Import MedViT từ code của bạn
from MedViT import MedViT_large
from ultrasound_dataset import UltrasoundTransform

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model paths
MEDVIT_CKPT_PATH = "models/MedViT_large_knee2.pth"
ULTRASAM_CONFIG_PATH = "configs/train_knee_5_class.py"
ULTRASAM_CKPT_PATH = "models/converted_best_coco_segm_mAP_iter_21200.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# UltraSAM class definitions
ULTRASAM_CLASSES = ['abnormal', 'fat', 'femur', 'tendon', 'fluid']
CLASSIFICATION_CLASSES = ['Viêm', 'Không viêm']

print("📄 Loading models...")

# Register all modules for UltraSAM
register_all_modules()

# Initialize EasyOCR reader
ocr_reader = None
try:
    print("📄 Loading OCR model...")
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

# Load UltraSAM Segmentation Model
ultrasam_model = None
try:
    if os.path.exists(ULTRASAM_CONFIG_PATH) and os.path.exists(ULTRASAM_CKPT_PATH):
        ultrasam_model = init_detector(ULTRASAM_CONFIG_PATH, ULTRASAM_CKPT_PATH, device=device)
        print("✅ UltraSAM Segmentation model loaded")
    else:
        print(f"❌ UltraSAM files not found:")
        print(f"Config: {ULTRASAM_CONFIG_PATH}")
        print(f"Checkpoint: {ULTRASAM_CKPT_PATH}")
except Exception as e:
    print(f"❌ Error loading UltraSAM: {e}")

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
        
        print(f"🔍 Image size: {width}x{height}, Header ROI: {width}x{header_height}")
        
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
        
        print(f"🔍 OCR found {len(results_all)} total detections")
        
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
        
        print(f"🔍 Unique texts: {unique_texts}")
        
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
    """Find ultrasound position from OCR text results"""
    
    # Define keyword sets
    ANATOMICAL_LOCATIONS = {
        'SUPRAPAT', 'SUPRAPATELLAR', 'SUPRA', 
        'POST', 'POSTERIOR',
        'LAT', 'LATERAL',
        'MED', 'MEDIAL', 
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
            'SUPRAPAT': 'suprapat', 'SUPRAPATELLAR': 'suprapat', 'SUPRA': 'suprapat',
            'POST': 'post', 'POSTERIOR': 'post',
            'LAT': 'lat', 'LATERAL': 'lat',
            'MED': 'med', 'MEDIAL': 'med',
            'ANT': 'ant', 'ANTERIOR': 'ant',
            'INFRAPATELLAR': 'infra', 'INFRA': 'infra'
        }
        return loc_map.get(loc, loc.lower())
    
    def normalize_orientation(ori):
        ori_map = {
            'LONG': 'long', 'LONGITUDINAL': 'long', 'LONGI': 'long',
            'TRANS': 'trans', 'TRANSVERSE': 'trans', 'TRANSV': 'trans'
        }
        return ori_map.get(ori, ori.lower())
    
    def normalize_side(side):
        side_map = {'LEFT': 'L', 'RIGHT': 'R', 'L': 'L', 'R': 'R'}
        return side_map.get(side, side)
    
    # Find keywords in texts
    found_locations = []
    found_orientations = []
    found_sides = []
    
    for text in texts:
        text_upper = text.upper().strip()
        
        if text_upper in ANATOMICAL_LOCATIONS:
            found_locations.append(normalize_location(text_upper))
        if text_upper in ORIENTATIONS:
            found_orientations.append(normalize_orientation(text_upper))
        if text_upper in SIDES:
            found_sides.append(normalize_side(text_upper))
    
    # Build position string
    if found_locations and found_orientations:
        location = found_locations[0]
        orientation = found_orientations[0]
        
        # Check for key combinations first
        key_position = f"{location}-{orientation}"
        if key_position in POSITION_ANATOMY:
            return key_position
        
        # If not a key position, return basic format
        if found_sides:
            position = f"{found_sides[0]} {location} {orientation}"
        else:
            position = f"{location} {orientation}"
        return position
    
    if found_locations:
        location = found_locations[0]
        if found_sides:
            position = f"{found_sides[0]} {location}"
        else:
            position = location
        return position
    
    return None

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
        class_label = CLASSIFICATION_CLASSES[pred_class]
        is_inflammation = (pred_class == 0)
        
        print(f"🔍 MedViT Classification: {class_label} (confidence: {confidence:.4f}, time: {infer_time:.2f}ms)")
        return is_inflammation, confidence, class_label
        
    except Exception as e:
        print(f"MedViT Classification error: {e}")
        return False, 0.5, "Lỗi phân loại MedViT"

def ultrasam_segment(image_path):
    """Perform segmentation using UltraSAM"""
    if ultrasam_model is None:
        return None, None, "UltraSAM model không khả dụng"
    
    try:
        start_time = time.perf_counter()
        
        # Run inference
        result = inference_detector(ultrasam_model, image_path)
        
        # Remove bboxes if present (we only need masks)
        if hasattr(result.pred_instances, 'bboxes'):
            delattr(result.pred_instances, 'bboxes')
        
        end_time = time.perf_counter()
        infer_time = (end_time - start_time) * 1000
        
        print(f"🔍 UltraSAM Segmentation completed (time: {infer_time:.2f}ms)")
        
        # Load original image
        img_rgb = mmcv.imread(image_path, channel_order='rgb')
        
        # Create visualizer
        vis = DetLocalVisualizer(line_width=2)
        vis.dataset_meta = {'classes': ULTRASAM_CLASSES}
        
        # Draw results
        vis.add_datasample(
            name='prediction',
            image=img_rgb.copy(),
            data_sample=result,
            draw_gt=False,
            pred_score_thr=0.5,
            show=False
        )
        visualized_image = vis.get_image()
        
        return result, visualized_image, None
        
    except Exception as e:
        print(f"UltraSAM Segmentation error: {e}")
        return None, None, f"Lỗi phân đoạn UltraSAM: {str(e)}"

def analyze_inflammation_from_segmentation(result, img_shape):
    """Analyze inflammation level from UltraSAM segmentation results"""
    if result is None or not hasattr(result, 'pred_instances'):
        return {
            'level': 0,
            'severity': 'Không xác định',
            'ratio': 0.0,
            'color': '#cccccc',
            'details': 'Không có kết quả phân đoạn'
        }
    
    try:
        pred_instances = result.pred_instances
        
        # Calculate total image area
        total_pixels = img_shape[0] * img_shape[1]
        
        # Count pixels for abnormal/inflammation class (class index 0)
        abnormal_pixels = 0
        
        if hasattr(pred_instances, 'masks') and hasattr(pred_instances, 'labels'):
            masks = pred_instances.masks
            labels = pred_instances.labels
            scores = pred_instances.scores if hasattr(pred_instances, 'scores') else None
            
            for i, label in enumerate(labels):
                # Check if this is abnormal class (index 0) with good confidence
                if label == 0 and (scores is None or scores[i] > 0.5):
                    mask = masks[i].cpu().numpy()
                    abnormal_pixels += np.sum(mask)
        
        # Calculate ratio
        ratio = (abnormal_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Determine inflammation level (3 levels as requested)
        if ratio > 15:  # High inflammation
            level = 3
            severity = "Nặng"
            color = "#ff4444"
        elif ratio > 5:  # Moderate inflammation
            level = 2
            severity = "Trung bình" 
            color = "#ff8800"
        elif ratio > 1:  # Mild inflammation
            level = 1
            severity = "Nhẹ"
            color = "#ffaa00"
        else:  # Very little or no inflammation
            level = 0
            severity = "Rất nhẹ/Không"
            color = "#44ff44"
        
        return {
            'level': level,
            'severity': severity,
            'ratio': round(ratio, 2),
            'color': color,
            'abnormal_pixels': int(abnormal_pixels),
            'total_pixels': int(total_pixels),
            'details': f'Tỉ lệ vùng bất thường: {ratio:.2f}% của toàn bộ ảnh'
        }
        
    except Exception as e:
        print(f"Error analyzing inflammation: {e}")
        return {
            'level': 0,
            'severity': 'Lỗi phân tích',
            'ratio': 0.0,
            'color': '#cccccc',
            'details': f'Lỗi: {str(e)}'
        }

def extract_segmentation_info(result):
    """Extract detailed segmentation information"""
    if result is None or not hasattr(result, 'pred_instances'):
        return {}
    
    try:
        pred_instances = result.pred_instances
        segmentation_info = {}
        
        if hasattr(pred_instances, 'masks') and hasattr(pred_instances, 'labels'):
            masks = pred_instances.masks
            labels = pred_instances.labels
            scores = pred_instances.scores if hasattr(pred_instances, 'scores') else None
            
            # Count objects by class
            class_counts = {}
            class_areas = {}
            
            for i, label in enumerate(labels):
                class_name = ULTRASAM_CLASSES[label.item()]
                confidence = scores[i].item() if scores is not None else 1.0
                
                # Only consider high confidence detections
                if confidence > 0.5:
                    mask = masks[i].cpu().numpy()
                    area = np.sum(mask)
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                        class_areas[class_name] = 0
                    
                    class_counts[class_name] += 1
                    class_areas[class_name] += area
            
            segmentation_info = {
                'detected_classes': class_counts,
                'class_areas': class_areas,
                'total_objects': sum(class_counts.values())
            }
        
        return segmentation_info
        
    except Exception as e:
        print(f"Error extracting segmentation info: {e}")
        return {}

@app.route('/')
def index():
    return render_template('index_ultrasam.html')

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    models = []
    
    if medvit_model is not None:
        models.append({
            'name': 'medvit',
            'display_name': 'MedViT Large',
            'description': 'Medical Vision Transformer - Specialized for medical imaging classification'
        })
    
    segmentation_models = []
    if ultrasam_model is not None:
        segmentation_models.append({
            'name': 'ultrasam',
            'display_name': 'UltraSAM',
            'description': 'Ultrasound Segmentation Model - Multi-class segmentation'
        })
    
    return jsonify({
        'success': True,
        'classification_models': models,
        'segmentation_models': segmentation_models,
        'available_classes': ULTRASAM_CLASSES
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Main API endpoint for image analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"🤖 Processing image: {file.filename}")
        
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image
        image_pil = Image.open(filepath)
        img_np = np.array(image_pil)
        
        # Ensure 3-channel image
        if img_np.ndim == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np
            
        H, W = img_3c.shape[:2]
        
        # Step 1: Extract position using OCR
        print("📤 Extracting position from image...")
        position = extract_position_from_image(image_pil)
        anatomy_info = get_position_anatomy_info(position)
        
        # Step 2: Classification with MedViT
        print("🔍 Running classification...")
        is_inflammation, confidence, class_label = classify_inflammation_medvit(image_pil)
        
        # Initialize result
        result = {
            'success': True,
            'filename': filename,
            'image_size': [W, H],
            'position': position,
            'anatomy_info': anatomy_info,
            'classification': {
                'has_inflammation': is_inflammation,
                'confidence': round(confidence * 100, 1),
                'label': class_label,
                'model': 'MedViT Large'
            },
            'segmentation': None,
            'inflammation_analysis': None
        }
        
        # Step 3: Segmentation with UltraSAM (always perform, regardless of classification)
        print("🎯 Running UltraSAM segmentation...")
        seg_result, visualized_image, seg_error = ultrasam_segment(filepath)
        
        if seg_result is not None and seg_error is None:
            # Analyze inflammation from segmentation
            inflammation_analysis = analyze_inflammation_from_segmentation(seg_result, (H, W))
            
            # Extract segmentation details
            segmentation_info = extract_segmentation_info(seg_result)
            
            result['segmentation'] = {
                'performed': True,
                'model': 'UltraSAM',
                'classes_detected': segmentation_info.get('detected_classes', {}),
                'total_objects': segmentation_info.get('total_objects', 0),
                'available_classes': ULTRASAM_CLASSES
            }
            result['inflammation_analysis'] = inflammation_analysis
            
            # Save result image with segmentation overlay
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            # Convert numpy array to PIL Image and save
            result_pil = Image.fromarray(visualized_image)
            result_pil.save(result_path)
            
        else:
            result['segmentation'] = {
                'performed': False,
                'error': seg_error or 'Unknown segmentation error'
            }
            
            # Save original image as result
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            image_pil.save(result_path)
        
        result['images'] = {
            'original': f"/uploads/{filename}",
            'result': f"/results/{result_filename}"
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'error': str(e)}), 500

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
            'ultrasam_segmentation': ultrasam_model is not None,
            'ocr_reader': ocr_reader is not None
        },
        'device': str(device),
        'available_classes': ULTRASAM_CLASSES
    })

if __name__ == '__main__':
    print("🚀 Medical Image Analysis API Server")
    print("📋 Features:")
    print("  - MedViT Classification (Inflammation detection)")
    print("  - UltraSAM Multi-class Segmentation")
    print("  - OCR Position Detection")
    print("  - 3-level Inflammation Analysis")
    print(f"🔗 http://localhost:5000")
    print(f"🔧 Health: http://localhost:5000/api/health")
    print(f"🤖 Available models: http://localhost:5000/api/models")
    app.run(host='0.0.0.0', port=5000, debug=True)