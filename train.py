import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

from model import AttentionCNN

# --- Configuration ---
DATA_DIR = 'dataset'
MODEL_SAVE_PATH = 'attention_model.pth'
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

class AttentionDataset(Dataset):
    """Custom Dataset for loading attention classification data."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        # Open image and convert to grayscale
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model():
    """
    Main function to train the attention classification model.
    """
    # --- 1. Data Augmentation and Loading ---

    # Define transformations for the training and validation sets
    # For training, we apply data augmentation to make the model more robust
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # For validation, we only need to resize and normalize
    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Define class names explicitly to ensure correct order
    class_names = ['attentive', 'distracted']

    # Load the datasets using ImageFolder
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')

    # Explicitly map class names to indices to avoid issues with extra folders
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    # Manually create the dataset to ignore unwanted folders
    def find_images_and_targets(dir, class_to_idx):
        images = []
        for target_class in class_to_idx.keys():
            class_dir = os.path.join(dir, target_class)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, fname)
                item = (path, class_to_idx[target_class])
                images.append(item)
        return images

    # Create datasets using our custom class
    train_dataset = AttentionDataset(find_images_and_targets(train_dir, class_to_idx), transform=train_transforms)
    val_dataset = AttentionDataset(find_images_and_targets(val_dir, class_to_idx), transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Found {len(train_dataset)} images in training set.")
    print(f"Found {len(val_dataset)} images in validation set.")
    print(f"Classes: {class_names}")

    # --- 2. Model, Loss, and Optimizer Initialization ---

    # Check for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = AttentionCNN(num_classes=len(class_names)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f'Epoch [{epoch+1}/{EPOCHS}] | Training Loss: {running_loss/len(train_loader):.4f} | Training Accuracy: {train_accuracy:.2f}%')

        # --- Validation Phase ---
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        print(f'Epoch [{epoch+1}/{EPOCHS}] | Validation Accuracy: {val_accuracy:.2f}%')

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'New best model saved to {MODEL_SAVE_PATH} with accuracy: {best_val_accuracy:.2f}%')

    print('\nFinished Training')

    # --- 4. Final Evaluation with Confusion Matrix ---
    print("\n--- Final Model Evaluation on Validation Set ---")
    # Load the best model for evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute and display the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print("Confusion matrix plot has been generated.")

if __name__ == '__main__':
    train_model()