import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
from PIL import Image

# Add backend to path to import architectures
sys.path.insert(0, str(Path(__file__).parent))
from backend.inference.models.architectures import AttentionNet, UNetLite

NUM_CLASSES = 10 # Adjust if your manifest uses 4 classes for everything!

class SyntheticBrainDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        label = np.random.randint(0, NUM_CLASSES)
        center_x = np.random.randint(50, self.img_size[1] - 50)
        center_y = np.random.randint(50, self.img_size[0] - 50)
        radius = np.random.randint(20, 40)
        
        y, x = np.ogrid[:self.img_size[0], :self.img_size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        color = np.random.rand(3)
        img[mask] = color
        
        img += np.random.normal(0, 0.1, img.shape)
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (2, 0, 1))
        
        seg_mask = mask.astype(np.float32)[np.newaxis, ...]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long), torch.from_numpy(seg_mask)

def train_model(model, dataloader, criterion_cls, criterion_seg=None, epochs=2):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels, masks in dataloader:
            optimizer.zero_grad()
            outputs = model(imgs)
            
            if isinstance(outputs, tuple):
                logits, seg_pred = outputs
                import torch.nn.functional as F
                masks_resized = F.interpolate(masks, size=seg_pred.shape[2:], mode='nearest')
                loss_cls = criterion_cls(logits, labels)
                if criterion_seg:
                    loss_seg = criterion_seg(seg_pred, masks_resized)
                    loss = loss_cls + 0.5 * loss_seg
                else:
                    loss = loss_cls
            else:
                loss = criterion_cls(outputs, labels)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
    
    return model

def main():
    print("Initializing Synthetic Dataset...")
    dataset = SyntheticBrainDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCELoss() 
    
    out_dir = Path("model_registry/weights")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nTraining Classifier (MobileNetV2)...")
    classifier = models.mobilenet_v2(weights=None)
    classifier.classifier[1] = nn.Linear(classifier.last_channel, 4)
    
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier.train()
    for epoch in range(2):
        loss_val = 0
        for imgs, labels, _ in dataloader:
            labels_4 = labels % 4
            optimizer.zero_grad()
            outputs = classifier(imgs)
            loss = criterion_cls(outputs, labels_4)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        print(f"  Epoch {epoch+1}/2 - Loss: {loss_val/len(dataloader):.4f}")
        
    cls_path = out_dir / "brain_mri_classifier_v1.pth"
    classifier.eval()
    torch.save(classifier, cls_path)
    print(f"Saved -> {cls_path}")
    
    print("\nTraining AttentionNet...")
    attention = AttentionNet(num_classes=10)
    attention = train_model(attention, dataloader, criterion_cls, criterion_seg=None, epochs=2)
    att_path = out_dir / "brain_mri_attention_v1.pth"
    attention.eval()
    torch.save(attention, att_path)
    print(f"Saved -> {att_path}")
    
    print("\nTraining UNetLite Segmenter...")
    segmenter = UNetLite(num_classes=10)
    segmenter = train_model(segmenter, dataloader, criterion_cls, criterion_seg, epochs=2)
    seg_path = out_dir / "brain_mri_segmenter_v1.pth"
    segmenter.eval()
    torch.save(segmenter, seg_path)
    print(f"Saved -> {seg_path}")
    
    print("\nTraining Complete! Restart backend to load new weights.")

if __name__ == '__main__':
    main()

