from torchvision.models import DenseNet121_Weights 
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  
import torch
import os


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print("GPU is available and will be used.")
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")
    return device

def freeze_layers(model, freeze_all=True):
    if freeze_all:
        print("Freezing all model layers")
        for param in model.parameters():
            param.requires_grad = False

def create_model(num_classes):
    print("Loading the pre-trained model...")
    base_model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1) 
    freeze_layers(base_model, freeze_all=True)

    print("Adding custom layers...")
    base_model.classifier = nn.Sequential(
        nn.Linear(base_model.classifier.in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )
    return base_model

def create_data_generators(train_dir, val_dir, input_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    prefetch_factor = 4
    train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count(), pin_memory=True, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=True, prefetch_factor=prefetch_factor)

    return train_loader, val_loader, train_dataset.classes


def main():
    name = "Dataset_2_cropped"
    train_dir = os.path.join(".", "dataset", f"{name}_split", "train")
    val_dir = os.path.join(".", "dataset", f"{name}_split", "validation")
    batch_size = 64
    input_size = (224, 224)
    num_epochs = 200
    patience = 10
    initial_learning_rate = 1e-3
    final_learning_rate = 1e-5

    device = set_device()

    print("Creating data generators...")
    train_loader, val_loader, class_labels = create_data_generators(train_dir, val_dir, input_size, batch_size)
    num_classes = len(class_labels)

    print(f"Class labels: {class_labels}")

    model = create_model(num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    def lr_lambda(epoch, num_epochs=num_epochs, lr_start=initial_learning_rate, lr_end=final_learning_rate):
        return (lr_end / lr_start) ** (epoch / (num_epochs - 1))
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print("Training the model...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

                # Update progress bar with the current training loss and accuracy
                pbar.set_postfix({'train_loss': train_loss / (len(train_loader.dataset)),
                                  'train_acc': train_correct / (len(train_loader.dataset))})
                pbar.update(1)  # Move the progress bar forward by 1 step

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        train_accuracy = train_correct / len(train_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, \
                Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving the model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('.', 'weights', f"{name}_best_densenet121.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()
