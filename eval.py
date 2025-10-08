import torch
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------
# Import model MedViT_tiny
# --------------------------
from MedViT import MedViT_tiny   # cần đúng đường dẫn file MedViT.py

# --------------------------
# Load model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MedViT_tiny(pretrained=False, num_classes=2).to(device)

model_path = "/home/coder/api/medsam_api/models/tiny-all.pth"
checkpoint = torch.load(model_path, map_location=device)

# Fix: Extract the actual model state_dict from the checkpoint
if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("MedViT_tiny model loaded successfully!")

# --------------------------
# DataLoader
# --------------------------
data_dir = "/home/coder/api/medsam_api/data_test"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

class_names = test_dataset.classes
# class_names = class_names_1[::-1]
print("Class indices:", test_dataset.class_to_idx)

# --------------------------
# Evaluation
# --------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Đảo ngược nhãn predictions: 0->1, 1->0
        preds = 1 - preds
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# --------------------------
# Metrics
# --------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="binary")
rec = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print("\n📊 Evaluation Results:")
print(f"Accuracy    : {acc:.4f}")
print(f"Precision   : {prec:.4f}")
print(f"Recall      : {rec:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"Specificity : {specificity:.4f}")
print("Confusion Matrix:\n", cm)

print("\nDetailed classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# --------------------------
# Save Confusion Matrix
# --------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_tiny.png")
plt.close()
print("Confusion matrix saved as confusion_matrix_tiny.png")