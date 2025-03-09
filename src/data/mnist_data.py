from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.seed import worker_init_fn, generator

class MNISTData:
    def __init__(self, batch_size=32, shuffle=True, data_dir="data"):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir

        self.transform = transforms.ToTensor()

        self.train_data = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        self.test_data = datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

        self.train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
        )
        self.test_dataloader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
        )

    def get_dataloaders(self):
        return self.train_dataloader, self.test_dataloader