import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import confusion_matrix
import numpy as np
from dataset3 import CustomImageDataset, custom_collate_fn, transform  # Adjust import statement as needed

# Load the trained model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.load_state_dict(torch.load("/home/adwait/data_set/data_set_armbench/vit_finetuned_best.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare the test dataloader
test_dataset = CustomImageDataset(
    csv_file='/home/adwait/data_set/data_set_armbench/test.csv',
    root_dir='/home/adwait/data_set/data_set_armbench/data',
    transform=transform  # Use the imported transform from dataset3
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=custom_collate_fn
)

# Evaluate metrics: Count, Recall, and FPR
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        predictions = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Calculate counts for each class
unique, counts = np.unique(all_labels, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"Counts per class: {class_counts}")

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Calculate percentages in confusion matrix
cm_percent = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100, 2)

# Define label classes
label_classes = ['multi_pick', 'nominal', 'package_defect']

# Print label classes
print(f"Label classes: {label_classes}")

# Print total number of images
print(f"Number of images in testing dataset: {len(test_dataset)}")

# Print counts per class
print(f"Counts per class: {class_counts}")

# Calculate recall for each class
recall = np.diag(cm) / np.sum(cm, axis=1)
print(f"Recall per class: {recall}")

# Calculate FPR for each class
fpr = []
for i in range(len(cm)):
    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
    fp = cm[:, i].sum() - cm[i, i]
    fpr.append(fp / (fp + tn))

print(f"False Positive Rate per class: {fpr}")

# Print confusion matrix with percentages
print("Confusion Matrix (Percentage):")
for i in range(len(label_classes)):
    row = [f"{value}%" for value in cm_percent[i]]
    print(f"{label_classes[i]}: {row}")

