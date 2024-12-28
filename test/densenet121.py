import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from train.densenet121 import create_model
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import torch


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and will be used.")
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")
    return device

def create_data_generator(test_dir, input_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    prefetch_factor = 2
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=True, prefetch_factor=prefetch_factor)

    return test_loader, test_dataset.classes

def load_model(model_path, num_classes, device):
    model = create_model(num_classes)  # Ensure this function matches the one in your training script
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Use weights_only=True
    model = model.to(device)
    model.eval()
    return model

def test_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    test_correct = 0
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()

                # Update progress bar with current test loss and accuracy
                pbar.set_postfix({'test_loss': test_loss / (test_correct + 1e-8),
                                  'test_acc': test_correct / (total_samples + 1e-8)})
                pbar.update(1)

    test_loss /= total_samples
    test_accuracy = test_correct / total_samples
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

def main():
    name = "Dataset_2_cropped"
    name2 = "test_inb_cropped"
    test_dir = os.path.join(".", "dataset", f"{name2}_split", "test")  # Directory for test data
    batch_size = 16
    input_size = (224, 224)

    device = set_device()

    print("Creating test data generator...")
    test_loader, class_labels = create_data_generator(test_dir, input_size, batch_size)
    num_classes = len(class_labels)

    print(f"Class labels: {class_labels}")

    model_path = os.path.join('.', 'weights', f"{name}_best_densenet121.pth")  # Path to the best saved model after training
    model = load_model(model_path, num_classes, device)

    print("Testing the model...")
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
