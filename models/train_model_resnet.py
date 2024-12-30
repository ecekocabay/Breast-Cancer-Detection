import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Benign, Malignant, Normal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/resnet18-f37072fd.pth"  # Path to ResNet18 weights


# Custom Dataset Class
class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Load data
def load_data(image_folder, labels_file):
    labels_df = pd.read_csv(labels_file)
    image_paths = [os.path.join(image_folder, f"{img}_results.jpg") for img in labels_df['Image Name']]
    labels = labels_df['Class'].map({'B': 0, 'M': 1, 'NORM': 2}).values
    return image_paths, labels


# Define ResNet18-based Model
class BreastCancerResNet(nn.Module):
    def __init__(self, num_classes, weights_path):
        super(BreastCancerResNet, self).__init__()
        self.model = models.resnet18()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

    return model


# Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['Benign', 'Malignant', 'Normal']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))


# Main Script
if __name__ == "__main__":
    # Load and preprocess data
    image_folder = "/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/output_images"
    labels_file = "/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/data/mias_labels_corrected.csv"
    image_paths, labels = load_data(image_folder, labels_file)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.3, random_state=42,
                                                        stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create datasets and dataloaders
    train_dataset = BreastCancerDataset(X_train, y_train, transform=transform)
    val_dataset = BreastCancerDataset(X_val, y_val, transform=transform)
    test_dataset = BreastCancerDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, criterion, and optimizer
    print("Loading ResNet18 model...")
    model = BreastCancerResNet(num_classes=NUM_CLASSES, weights_path=WEIGHTS_PATH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Training the model...")
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS, DEVICE)

    # Save the model
    model_path = "breast_cancer_resnet18.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(trained_model, test_loader, DEVICE)