import os
import pandas as pd
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_series = pd.read_csv(csv_file, header=None)[0].tolist()
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.images = []
        self.labels = []
        self._prepare_label_encoder()
        self._prepare_image_label_list()

    def _prepare_label_encoder(self):
        all_labels = set()
        for img_series in self.image_series:
            img_folder = os.path.join(self.root_dir, img_series)
            json_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.json')])
            for json_file in json_files:
                json_path = os.path.join(img_folder, json_file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        all_labels.add(data['label'])
                except Exception as e:
                    print(f"Error reading JSON file {json_path}: {e}")
                    continue

        self.label_encoder.fit(list(all_labels))
        print(f"Label classes: {self.label_encoder.classes_}")

    def _prepare_image_label_list(self):
        for img_series in self.image_series:
            img_folder = os.path.join(self.root_dir, img_series)
            image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])
            json_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.json')])

            for img_file, json_file in zip(image_files, json_files):
                img_path = os.path.join(img_folder, img_file)
                json_path = os.path.join(img_folder, json_file)

                self.images.append(img_path)
                self.labels.append(json_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        json_path = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                label = data['label']
                encoded_label = self.label_encoder.transform([label])[0]
        except Exception as e:
            print(f"Error reading JSON file {json_path}: {e}")
            return None, None

        return image, encoded_label

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None

    images, labels = zip(*batch)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(csv_file='/home/adwait/data_set/data_set_armbench/train.csv', root_dir='/home/adwait/data_set/data_set_armbench/data', transform=transform)
test_dataset = CustomImageDataset(csv_file='/home/adwait/data_set/data_set_armbench/test.csv', root_dir='/home/adwait/data_set/data_set_armbench/data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

def validate_data_loader(loader, dataset_name):
    print(f"Starting data validation for {dataset_name} data loader...")
    valid_samples = 0
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        images, labels = batch
        print(f"Sample {i}: Images shape: {images.shape}, Labels: {labels}")
        valid_samples += 1
        if valid_samples == 5:
            break

validate_data_loader(train_loader, "Training")
validate_data_loader(test_loader, "Test")


num_train_images = len(train_dataset)
num_test_images = len(test_dataset)

# Print the counts
print(f"Number of images in training dataset: {num_train_images}")
print(f"Number of images in testing dataset: {num_test_images}")

