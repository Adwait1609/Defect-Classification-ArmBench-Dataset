import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import recall_score, roc_curve
from tqdm import tqdm

from dataset import CustomImageDataset  # Import your dataset class here

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensors
])

# Load validation dataset
val_dataset = CustomImageDataset(csv_file='/home/adwait/data_set/data_set_armbench/test.csv', root_dir='/home/adwait/data_set/data_set_armbench/data', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model setup (ViT)
vit_model = models.vit_b_16(weights=None).to(device)  # Initialize the model without pretrained weights
vit_model.load_state_dict(torch.load('/home/adwait/data_set/data_set_armbench/model.pth'))  # Load the trained model state dictionary

# Evaluation function
def validate_model():
    vit_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = vit_model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    recall = recall_score(all_labels, all_preds, average=None)
    fpr = [roc_curve(all_labels, all_preds, pos_label=i)[0] for i in range(len(val_dataset.label_encoder.classes_))]
    
    # Print results
    print("Validation Metrics:")
    print(f"Recall: {recall}")
    print(f"False Positive Rate: {fpr}")

validate_model()
print("Validation completed.")
