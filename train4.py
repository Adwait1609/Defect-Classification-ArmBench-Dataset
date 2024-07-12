# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from dataset3 import CustomImageDataset, custom_collate_fn, transform  # Adjust import statement as needed

# Load the pretrained model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the training parameters
learning_rate = 1e-5
num_epochs = 100  # Start with a large number of epochs
batch_size = 32
patience = 10  # Number of epochs with no improvement after which training will be stopped
best_loss = float('inf')
epochs_no_improve = 0

# Prepare the dataloaders
train_dataset = CustomImageDataset(
    csv_file='/home/adwait/data_set/data_set_armbench/train.csv',
    root_dir='/home/adwait/data_set/data_set_armbench/data',
    transform=transform  # Use the imported transform from dataset3
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate_fn
)

test_dataset = CustomImageDataset(
    csv_file='/home/adwait/data_set/data_set_armbench/test.csv',
    root_dir='/home/adwait/data_set/data_set_armbench/data',
    transform=transform  # Use the imported transform from dataset3
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=custom_collate_fn
)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate loss
        optimizer.step()  # Update parameters
        running_loss += loss.item()

    # Print average loss for this epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Validate the model
    model.eval()
    total_correct = 0
    total_samples = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits  # Forward pass
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    val_loss /= len(test_loader)
    print(f"Validation Accuracy: {accuracy * 100}%, Validation Loss: {val_loss}")

    # Check if validation loss has improved
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        # Save the model if it is the best seen so far
        torch.save(model.state_dict(), "/home/adwait/data_set/data_set_armbench/vit_finetuned_best.pth")
    else:
        epochs_no_improve += 1
        # Check if we have reached patience
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
