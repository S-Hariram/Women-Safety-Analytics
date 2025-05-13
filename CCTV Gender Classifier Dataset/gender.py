import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torch.multiprocessing

torch.multiprocessing.set_start_method('spawn', force=True)

# ✅ Paths
BASE_PATH = r"C:\\sheild-master\\CCTV Gender Classifier Dataset"
TRAIN_PATH = BASE_PATH  

# ✅ Define Classes Manually (Only Male & Female)
CLASSES = ["FEMALE", "MALE"]
NUM_CLASSES = len(CLASSES)
print(f"✅ Classes: {CLASSES}")

# ✅ Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Training on: {device}")

# ✅ Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Custom Dataset Class
class GenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(CLASSES):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_path):  # Ensure folders exist
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ✅ Model (Using MobileNetV3 for Efficiency)
class GenderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GenderClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet.classifier[3] = nn.Linear(1280, num_classes)  
    
    def forward(self, x):
        return self.mobilenet(x)

if __name__ == "__main__":
    # ✅ Load Dataset & Split (80% Train, 20% Test)
    full_dataset = GenderDataset(TRAIN_PATH, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # ✅ Initialize Model
    model = GenderClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Training Loop
    num_epochs = 50  
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"✅ Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # ✅ Save Model
    torch.save(model.state_dict(), os.path.join(BASE_PATH, "gender_classifier.pth"))
    print("✅ Model Saved!")

    # ✅ Testing Loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Model Accuracy: {accuracy:.2f}%")
