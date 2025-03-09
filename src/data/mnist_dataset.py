import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
from PIL import Image
import pathlib

from src.utils.seed import worker_init_fn, generator

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        self.transform = Compose([
            ToTensor(),
        ])

        self.image_paths = [str(p) for p in pathlib.Path(dir_path).glob("**/*.jpg")]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        p = self.image_paths[index]
        path = pathlib.Path(p)
        image = Image.open(path)

        if self.transform:
            X = self.transform(image)

        y = int(str(path.parent.name))
        return X, y

def create_dataset(train_path, test_path, batch_size):
    train_dataset = MNISTDataset(train_path)
    test_dataset = MNISTDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    
    return train_loader, test_loader