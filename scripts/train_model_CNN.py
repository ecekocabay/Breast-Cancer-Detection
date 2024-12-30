import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Constants
IMG_SIZE = (224, 224)  # Input image size
FEATURE_SHAPE = (3, 224, 224)  # Channels, Height, Width
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Benign, Malignant, Normal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BreastCancerDataset(Dataset):
    def __init__(self, features, labels, feature_shape):
        """
        Dataset class for numerical features and labels.

        :param features: Numerical feature matrix.
        :param labels: Corresponding labels.
        :param feature_shape: Shape to reshape the features into (channels, height, width).
        """
        self.features = features
        self.labels = labels
        self.feature_shape = feature_shape

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].reshape(self.feature_shape)  # Reshape to (channels, height, width)
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float), torch.tensor(label, dtype=torch.long)


# Load data
def load_data(image_folder, labels_file):
    labels_df = pd.read_csv(labels_file)
    image_paths = [os.path.join(image_folder, f"{img}_results.jpg") for img in labels_df['Image Name']]
    labels = labels_df['Class'].map({'B': 0, 'M': 1, 'NORM': 2}).values  # Encode labels as integers
    return image_paths, labels


# Extract features from images
def extract_features(image_paths, transform):
    """
    Extract features for all images using the given transformations.
    """
    features = []
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        features.append(image.numpy())  # Convert to numpy array
    return np.array(features)


# Define CNN Model
class BreastCancerCNN(nn.Module):
    def __init__(self, num_classes):
        super(BreastCancerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return model


# Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['Benign', 'Malignant', 'Normal']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

# Save the trained model
def save_model(model, path):
    """
    Save the trained model to a file.

    :param model: Trained PyTorch model.
    :param path: Path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Load a saved model
def load_model(model_class, path, device, num_classes=NUM_CLASSES):
    """
    Load a saved model from a file.

    :param model_class: Model class (e.g., BreastCancerCNN).
    :param path: Path to the saved model.
    :param device: Device to load the model onto (CPU or GPU).
    :param num_classes: Number of classes for the model.
    :return: Loaded model.
    """
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model


# Main script
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

    # Extract features for SMOTE
    print("Extracting features for SMOTE...")
    X_train_features = extract_features(X_train, transform)
    X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)  # Flatten features

    # Apply SMOTE to balance classes
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)

    # Create datasets and dataloaders
    print("Creating datasets...")
    train_dataset = BreastCancerDataset(
        X_train_resampled.reshape(-1, *FEATURE_SHAPE), y_train_resampled, feature_shape=FEATURE_SHAPE
    )
    val_dataset = BreastCancerDataset(extract_features(X_val, transform).reshape(len(X_val), -1), y_val, FEATURE_SHAPE)
    test_dataset = BreastCancerDataset(extract_features(X_test, transform).reshape(len(X_test), -1), y_test, FEATURE_SHAPE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # Initialize model, criterion, and optimizer
    model = BreastCancerCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Training the model...")
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS, DEVICE)

    # Save the trained model
    model_path = "/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/models/breast_cancer_cnn_model.pth"
    save_model(trained_model, model_path)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(trained_model, test_loader, DEVICE)

    # Example: Load the model for future use
    loaded_model = load_model(BreastCancerCNN, model_path, DEVICE)
    print("Model reloaded successfully.")