import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def prepare_dataset(batch_size: int = 32):
    transform = transforms.Compose([
        transforms.Resize((44, 44)),  # Resize to the input size of the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SFEW('./data/SFEW/Train', transform=transform)
    val_dataset = SFEW('./data/SFEW/Val', transform=transform)
    # test_dataset = SFEW('SFEW/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, None


class SFEW(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.classes = os.listdir(directory)
        self.images = []
        self.labels = []

        for index, label in enumerate(self.classes):
            class_path = os.path.join(directory, label)
            for img_file in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_file))
                self.labels.append(index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
