import os
import torch
import numpy as np
from PIL import Image
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import json
from itertools import chain

# Import từ code của bạn
from MedViT import MedViT_large
from ultrasound_dataset import UltrasoundTransform

class ModelTester:
    def __init__(self, 
                 medvit_path="models/MedViT_large_knee2.pth",
                 efficientnet_path="models/efficientnet_b0_ultrasound_2_class.pth",
                 device=None):
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['Viêm', 'Không viêm']
        
        print(f"🔧 Device: {self.device}")
        
        # Load MedViT
        self.medvit_model = None
        self.medvit_transform = None
        try:
            self.medvit_model = MedViT_large(num_classes=2).to(self.device)
            if os.path.exists(medvit_path):
                checkpoint = torch.load(medvit_path, map_location=self.device, weights_only=False)
                self.medvit_model.load_state_dict(checkpoint['model'], strict=False)
                self.medvit_model.eval()
                self.medvit_transform = UltrasoundTransform(is_training=False)
                print("✅ MedViT model loaded successfully")
            else:
                print(f"❌ MedViT checkpoint not found: {medvit_path}")
        except Exception as e:
            print(f"❌ Error loading MedViT: {e}")
        
        # Load EfficientNet
        self.efficientnet_model = None
        self.efficientnet_transform = None
        try:
            self.efficientnet_model = models.efficientnet_b0(weights=None)  # Updated deprecated parameter
            self.efficientnet_model.classifier[1] = torch.nn.Linear(
                self.efficientnet_model.classifier[1].in_features, 2
            )
            self.efficientnet_model = self.efficientnet_model.to(self.device)
            
            if os.path.exists(efficientnet_path):
                checkpoint = torch.load(efficientnet_path, map_location=self.device, weights_only=False)
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    self.efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    self.efficientnet_model.load_state_dict(checkpoint['model'])
                else:
                    self.efficientnet_model.load_state_dict(checkpoint)
                
                self.efficientnet_model.eval()
                
                # EfficientNet preprocessing
                self.efficientnet_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("✅ EfficientNet model loaded successfully")
            else:
                print(f"❌ EfficientNet checkpoint not found: {efficientnet_path}")
        except Exception as e:
            print(f"❌ Error loading EfficientNet: {e}")
    
    def load_dataset(self, inflammation_dir, no_inflammation_dir):
        """Load dataset từ 2 thư mục"""
        data = []
        labels = []
        
        # Supported image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        
        # Load ảnh viêm (label = 0)
        inflammation_path = Path(inflammation_dir)
        if inflammation_path.exists():
            inflammation_files = []
            for ext in image_extensions:
                inflammation_files.extend(inflammation_path.glob(ext))
            
            print(f"📁 Found {len(inflammation_files)} inflammation image files")
            
            for img_file in inflammation_files:
                try:
                    img = Image.open(img_file)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    data.append(img)
                    labels.append(0)  # Viêm
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            print(f"✅ Loaded {len([l for l in labels if l == 0])} inflammation images")
        else:
            print(f"❌ Inflammation directory not found: {inflammation_dir}")
        
        # Load ảnh không viêm (label = 1)
        no_inflammation_path = Path(no_inflammation_dir)
        if no_inflammation_path.exists():
            inflammation_count = len([l for l in labels if l == 0])  # Count inflammation images
            no_inflammation_files = []
            for ext in image_extensions:
                no_inflammation_files.extend(no_inflammation_path.glob(ext))
            
            print(f"📁 Found {len(no_inflammation_files)} no-inflammation image files")
            
            for img_file in no_inflammation_files:
                try:
                    img = Image.open(img_file)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    data.append(img)
                    labels.append(1)  # Không viêm
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            no_inflammation_count = len([l for l in labels if l == 1])
            print(f"✅ Loaded {no_inflammation_count} no-inflammation images")
        else:
            print(f"❌ No-inflammation directory not found: {no_inflammation_dir}")
        
        print(f"📊 Total dataset: {len(data)} images")
        print(f"   - Viêm: {labels.count(0)} images")
        print(f"   - Không viêm: {labels.count(1)} images")
        
        return data, labels
    
    @torch.no_grad()
    def predict_medvit(self, image):
        """Predict với MedViT"""
        if self.medvit_model is None:
            return None, None, None
        
        try:
            image_tensor = self.medvit_transform(image).unsqueeze(0).to(self.device)
            
            start_time = time.perf_counter()
            output = self.medvit_model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # ms
            return pred_class, confidence, inference_time
            
        except Exception as e:
            print(f"MedViT prediction error: {e}")
            return None, None, None
    
    @torch.no_grad()
    def predict_efficientnet(self, image):
        """Predict với EfficientNet"""
        if self.efficientnet_model is None:
            return None, None, None
        
        try:
            image_tensor = self.efficientnet_transform(image).unsqueeze(0).to(self.device)
            
            start_time = time.perf_counter()
            output = self.efficientnet_model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # ms
            return pred_class, confidence, inference_time
            
        except Exception as e:
            print(f"EfficientNet prediction error: {e}")
            return None, None, None
    
    def evaluate_model(self, data, labels, model_name):
        """Evaluate một model"""
        predictions = []
        confidences = []
        inference_times = []
        
        print(f"\n🔄 Testing {model_name}...")
        
        for i, image in enumerate(data):
            if model_name.lower() == 'medvit':
                pred, conf, time_ms = self.predict_medvit(image)
            elif model_name.lower() == 'efficientnet':
                pred, conf, time_ms = self.predict_efficientnet(image)
            else:
                continue
            
            if pred is not None:
                predictions.append(pred)
                confidences.append(conf)
                inference_times.append(time_ms)
            
            # Progress
            if (i + 1) % 10 == 0 or i == len(data) - 1:
                print(f"   Progress: {i+1}/{len(data)} images processed")
        
        if len(predictions) == 0:
            print(f"❌ No predictions made for {model_name}")
            return None
        
        # Make sure we have the same number of labels as predictions
        if len(predictions) != len(labels):
            print(f"⚠️  Warning: {len(predictions)} predictions but {len(labels)} labels")
            # Take only the first len(predictions) labels
            labels = labels[:len(predictions)]
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Average inference time
        avg_time = np.mean(inference_times)
        
        results = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'avg_inference_time_ms': avg_time,
            'predictions': predictions,
            'confidences': confidences,
            'inference_times': inference_times
        }
        
        return results
    
    def compare_models(self, inflammation_dir, no_inflammation_dir, save_results=True):
        """So sánh 2 models"""
        print("🚀 Starting model comparison...")
        
        # Load dataset
        data, labels = self.load_dataset(inflammation_dir, no_inflammation_dir)
        
        if len(data) == 0:
            print("❌ No data loaded!")
            return
        
        results = {}
        
        # Test MedViT
        if self.medvit_model is not None:
            medvit_results = self.evaluate_model(data, labels, 'MedViT')
            if medvit_results:
                results['MedViT'] = medvit_results
        
        # Test EfficientNet
        if self.efficientnet_model is not None:
            efficientnet_results = self.evaluate_model(data, labels, 'EfficientNet')
            if efficientnet_results:
                results['EfficientNet'] = efficientnet_results
        
        # Print comparison
        self.print_comparison(results)
        
        # Plot results
        if len(results) > 0:
            self.plot_results(results, labels)
        
        # Save results
        if save_results and len(results) > 0:
            self.save_results(results)
        
        return results
    
    def print_comparison(self, results):
        """In kết quả so sánh"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n🤖 {model_name}:")
            print(f"   Accuracy:  {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print(f"   Precision: {result['precision']:.4f}")
            print(f"   Recall:    {result['recall']:.4f}")
            print(f"   F1-Score:  {result['f1_score']:.4f}")
            print(f"   Avg Time:  {result['avg_inference_time_ms']:.2f} ms")
            
            # Confusion matrix
            cm = np.array(result['confusion_matrix'])
            print(f"   Confusion Matrix:")
            print(f"   True\\Pred  Viêm  Không viêm")
            print(f"   Viêm       {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"   Không viêm {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # So sánh tổng quan
        if len(results) >= 2:
            print(f"\n🏆 WINNER COMPARISON:")
            models = list(results.keys())
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                values = [(name, results[name][metric]) for name in models]
                winner = max(values, key=lambda x: x[1])
                print(f"   {metric.capitalize():12}: {winner[0]} ({winner[1]:.4f})")
            
            # Speed comparison
            speed_values = [(name, results[name]['avg_inference_time_ms']) for name in models]
            fastest = min(speed_values, key=lambda x: x[1])
            print(f"   {'Speed':12}: {fastest[0]} ({fastest[1]:.2f} ms)")
    
    def plot_results(self, results, true_labels):
        """Vẽ biểu đồ kết quả"""
        try:
            fig, axes = plt.subplots(2, len(results), figsize=(6*len(results), 12))
            if len(results) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, (model_name, result) in enumerate(results.items()):
                # Confusion Matrix
                cm = np.array(result['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Viêm', 'Không viêm'],
                           yticklabels=['Viêm', 'Không viêm'],
                           ax=axes[0, i])
                axes[0, i].set_title(f'{model_name} - Confusion Matrix')
                axes[0, i].set_ylabel('True Label')
                axes[0, i].set_xlabel('Predicted Label')
                
                # Metrics comparison
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                values = [result['accuracy'], result['precision'], result['recall'], result['f1_score']]
                
                bars = axes[1, i].bar(metrics, values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
                axes[1, i].set_title(f'{model_name} - Metrics')
                axes[1, i].set_ylim(0, 1)
                axes[1, i].set_ylabel('Score')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
            print("📊 Plot saved: model_comparison_results.png")
            
            # Inference time comparison
            if len(results) >= 2:
                plt.figure(figsize=(8, 6))
                models = list(results.keys())
                times = [results[model]['avg_inference_time_ms'] for model in models]
                
                bars = plt.bar(models, times, color=['#FF6B6B', '#4ECDC4'])
                plt.title('Average Inference Time Comparison')
                plt.ylabel('Time (ms)')
                plt.xlabel('Model')
                
                # Add value labels
                for bar, time_val in zip(bars, times):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{time_val:.2f}ms', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('inference_time_comparison.png', dpi=300, bbox_inches='tight')
                print("📊 Plot saved: inference_time_comparison.png")
                
        except Exception as e:
            print(f"❌ Error creating plots: {e}")
    
    def save_results(self, results):
        """Save kết quả ra file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model_name, result in results.items():
                json_result = result.copy()
                json_result['confusion_matrix'] = np.array(result['confusion_matrix']).tolist()
                # Remove non-serializable data
                json_result.pop('predictions', None)
                json_result.pop('confidences', None)
                json_result.pop('inference_times', None)
                json_results[model_name] = json_result
            
            # Save to JSON
            with open('model_comparison_results.json', 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to 'model_comparison_results.json'")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function để chạy test"""
    # Đường dẫn đến thư mục chứa ảnh
    inflammation_dir = input("📁 Nhập đường dẫn thư mục chứa ảnh VIÊM: ").strip()
    no_inflammation_dir = input("📁 Nhập đường dẫn thư mục chứa ảnh KHÔNG VIÊM: ").strip()
    
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(inflammation_dir):
        print(f"❌ Thư mục không tồn tại: {inflammation_dir}")
        return
    
    if not os.path.exists(no_inflammation_dir):
        print(f"❌ Thư mục không tồn tại: {no_inflammation_dir}")
        return
    
    # Tạo tester
    tester = ModelTester(
        medvit_path="models/MedViT_large_knee2.pth",
        efficientnet_path="models/efficientnet_b0_ultrasound_2_class.pth"
    )
    
    # Chạy comparison
    results = tester.compare_models(inflammation_dir, no_inflammation_dir)
    
    print("\n✅ Testing completed!")
    
    return results


if __name__ == "__main__":
    results = main()