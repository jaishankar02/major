import os
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CityScapeDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256,256))

        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)

        img = torch.tensor(img).permute(2,0,1).float()/255.0
        mask = torch.tensor(mask).long()

        return img, mask


def get_loaders(data_dir, batch_size=8):
    img_dir = os.path.join(data_dir, "CameraRGB")
    mask_dir = os.path.join(data_dir, "CameraMask")

    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    img_paths = [os.path.join(img_dir, x) for x in images]
    mask_paths = [os.path.join(mask_dir, x) for x in masks]

    train_img, test_img, train_mask, test_mask = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_ds = CityScapeDataset(train_img, train_mask)
    test_ds = CityScapeDataset(test_img, test_mask)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader