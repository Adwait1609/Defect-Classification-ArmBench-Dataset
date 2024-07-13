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
        self._prepare_label_encoder()

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

    def __len__(self):
        return len(self.image_series)

    def count_images(self):
        count = 0
        for img_series in self.image_series:
            img_folder = os.path.join(self.root_dir, img_series)
            image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg') and f.startswith(img_series)])
            count += len(image_files)
        return count

    def __getitem__(self, idx):
        img_series = str(self.image_series[idx])
        img_folder = os.path.join(self.root_dir, img_series)
        
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg') and f.startswith(img_series)], 
                             key=lambda x: int(x.split('_')[-1].replace('.jpg', '')))
        json_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.json') and f.startswith(img_series)], 
                            key=lambda x: int(x.split('_')[-1].replace('.json', '')))
        
        if len(image_files) == 0 or len(json_files) == 0:
            print(f"Skipping series {img_series} due to missing files.")
            return None, None
        
        images = []
        json_data = []

        for i in range(len(image_files)):
            img_file = image_files[i]
            json_file = json_files[i]

            img_path = os.path.join(img_folder, img_file)
            json_path = os.path.join(img_folder, json_file)

            if not os.path.exists(img_path):
                print(f"Image file not found: {img_path}")
                return None, None

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None, None

            images.append(image)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    json_data.append(data)
            except Exception as e:
                print(f"Error reading JSON file {json_path}: {e}")
                return None, None

        if self.transform:
            images = [self.transform(img) for img in images]

        labels = [data['label'] for data in json_data]
        encoded_labels = self.label_encoder.transform(labels)

        return images, encoded_labels

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None

    images, labels = zip(*batch)

    # Determine maximum sequence length
    max_len = max([len(img_seq) for img_seq in images])

    # Create padded tensor for images
    padded_images = torch.zeros(len(images), max_len, images[0][0].size(0), images[0][0].size(1), images[0][0].size(2))

    # Fill in the padded tensor with actual images
    for i, img_seq in enumerate(images):
        seq_len = len(img_seq)
        for j in range(seq_len):
            padded_images[i, j, :, :] = img_seq[j]

    # Create padded tensor for labels
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)  # Assuming labels are long type

    for i, label_seq in enumerate(labels):
        seq_len = len(label_seq)
        padded_labels[i, :seq_len] = torch.tensor(label_seq)

    return padded_images, padded_labels



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



# Count the number of images in the training dataset
train_image_count = train_dataset.count_images()
#print(f"Number of images in training dataset: {train_image_count}")

# Count the number of images in the test dataset
test_image_count = test_dataset.count_images()
#print(f"Number of images in test dataset: {test_image_count}")








   

















