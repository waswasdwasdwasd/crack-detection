# crack_detection.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from pathlib import Path
import random
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# CELL 2: Load Local Dataset (no need run if have the files already)
#import zipfile

# Just provide the path to your ZIP file
#zip_path = r"C:\Users\alanb\Downloads\archive (2).zip"  # Change this to your path

# Extract ZIP
#print(f"📦 Extracting {zip_path}...")
#with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#    zip_ref.extractall('uploaded_dataset')

# Find folders
#all_folders = [f for f in os.listdir('uploaded_dataset') if os.path.isdir(os.path.join('uploaded_dataset', f))]
#all_folders = [f for f in all_folders if not f.startswith('__')]

#print(f"\n✅ Found folders:")
#for i, folder in enumerate(all_folders, 1):
    #folder_path = os.path.join('uploaded_dataset', folder)
    #num_files = len([f for f in Path(folder_path).glob('**/*') if f.is_file()])
    #print(f"   {i}. {folder} ({num_files} files)")

# Set folder names
#positive_folder = "Positive"  # Change if different
#negative_folder = "Negative"  # Change if different

#positive_path = os.path.join('uploaded_dataset', positive_folder)
#negative_path = os.path.join('uploaded_dataset', negative_folder)

# Create organized dataset structure
# os.makedirs('data/train/crack', exist_ok=True)
# os.makedirs('data/train/no_crack', exist_ok=True)
# os.makedirs('data/val/crack', exist_ok=True)
# os.makedirs('data/val/no_crack', exist_ok=True)

# def process_folder(folder_path, label_type, max_images=20000, split_ratio=0.8):
#     """Process images from folder and split into train/val"""
#     image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
#     all_images = []
    
#     for ext in image_extensions:
#         all_images.extend(Path(folder_path).glob(f'**/*{ext}'))
#         all_images.extend(Path(folder_path).glob(f'**/*{ext.upper()}'))
    
#     all_images = list(all_images)[:max_images]
#     print(f"   Found {len(all_images)} images")
    
#     random.seed(42)
#     random.shuffle(all_images)
    
#     split_idx = int(len(all_images) * split_ratio)
#     train_images = all_images[:split_idx]
#     val_images = all_images[split_idx:]
    
#     print(f"   Organizing {len(train_images)} training images...")
#     for idx, img_path in enumerate(train_images):
#         img = Image.open(img_path).convert('RGB')
#         img.save(f'data/train/{label_type}/{idx}.jpg')
    
#     print(f"   Organizing {len(val_images)} validation images...")
#     for idx, img_path in enumerate(val_images):
#         img = Image.open(img_path).convert('RGB')
#         img.save(f'data/val/{label_type}/{idx}.jpg')
    
#     return len(train_images), len(val_images)

# print("\n📥 Processing positive images...")
# pos_train, pos_val = process_folder(positive_path, 'crack')

# print("\n📥 Processing negative images...")
# neg_train, neg_val = process_folder(negative_path, 'no_crack')

# print(f"\n✅ DATASET READY!")
# print(f"   Train: {pos_train} crack, {neg_train} no-crack")
# print(f"   Val: {pos_val} crack, {neg_val} no-crack")

# CELL 3: Dataset and DataLoader
class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load crack images (label=1)
        crack_dir = os.path.join(root_dir, 'crack')
        for img_name in os.listdir(crack_dir):
            self.images.append(os.path.join(crack_dir, img_name))
            self.labels.append(1)
        
        # Load no-crack images (label=0)
        no_crack_dir = os.path.join(root_dir, 'no_crack')
        for img_name in os.listdir(no_crack_dir):
            self.images.append(os.path.join(no_crack_dir, img_name))
            self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CrackDataset('data/train', transform=train_transform)
val_dataset = CrackDataset('data/val', transform=val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"📊 Dataloaders ready:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# CELL 4: Create Model
# Using ResNet18 (fast and accurate for binary classification)
model = models.resnet18(weights='IMAGENET1K_V1')

# Modify last layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 2)  # 2 classes: crack, no_crack
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("✅ Model created: ResNet18 for binary crack classification")
print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# CELL 5: Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5):
    """Train the crack classifier"""
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\n📍 Epoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_crack_model.pth')
            print(f'✨ New best model saved! Acc: {best_acc:.4f}')
        
        scheduler.step()
    
    return model, history

# Train the model
model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5)

# CELL 6: Comprehensive Metrics
def evaluate_model(model, val_loader):
    """Calculate all metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*50)
    print("📊 MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"✓ Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✓ Precision:  {precision:.4f}")
    print(f"✓ Recall:     {recall:.4f}")
    print(f"✓ F1-Score:   {f1:.4f}")
    print(f"✓ Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    print("="*50)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Crack', 'Crack'],
                yticklabels=['No Crack', 'Crack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Over Time')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return accuracy, precision, recall, f1

# Load best model and evaluate
model.load_state_dict(torch.load('best_crack_model.pth'))
metrics = evaluate_model(model, val_loader)

# CELL 7: Grad-CAM Visualization (Shows WHERE the crack is)
def visualize_crack_location(model, image_path, save_path=None):
    """Use Grad-CAM to show crack location"""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    
    # Transform for model
    input_tensor = val_transform(Image.fromarray(img_resized)).unsqueeze(0).to(device)
    
    # Setup Grad-CAM
    target_layer = model.layer4[-1]  # Last conv layer of ResNet
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay heatmap on image
    cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
    
    # Get prediction and confidence
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0, 1].item()  # Probability of crack
        pred_class = 'CRACK' if confidence > 0.5 else 'NO CRACK'
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(img_resized)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(cam_image)
    ax2.set_title(f'Crack Detection Heatmap\n{pred_class} (Confidence: {confidence:.3f})')
    ax2.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return pred_class, confidence

# Test on sample images
print("🔍 Testing Grad-CAM visualization on validation images...")
sample_crack = val_dataset.images[0]  # First crack image
sample_no_crack = val_dataset.images[len(os.listdir('data/val/crack'))]  # First no-crack

visualize_crack_location(model, sample_crack)
visualize_crack_location(model, sample_no_crack)

# CELL 8: Real-Time Webcam Detection (VSCode version)
def webcam_crack_detection_vscode(model):
    """Real-time crack detection with webcam"""
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for built-in, 1 for external webcam
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return
    
    print("📹 Webcam started!")
    print("Press 'SPACE' to analyze frame")
    print("Press 'Q' to quit\n")
    
    model.eval()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display live feed
        cv2.imshow('Crack Detection - Press SPACE to analyze', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Press SPACE to analyze
        if key == ord(' '):
            # Save frame temporarily
            cv2.imwrite('temp_frame.jpg', frame)
            
            # Analyze with Grad-CAM
            pred_class, confidence = visualize_crack_location(model, 'temp_frame.jpg')
            
            print(f"🎯 Result: {pred_class} (Confidence: {confidence:.1%})")
        
        # Press Q to quit
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam closed")

# Run webcam detection
webcam_crack_detection_vscode(model)
