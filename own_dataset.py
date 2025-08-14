import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from torchvision import transforms


class MetaHumanHairMattingDataset(Dataset):
    def __init__(self, path='/datasets/generated/metahuman_matting_second_test/', resize=(512, 512)):
        self.path = path
        self.resize = resize
        self.samples = pd.read_csv(os.path.join(path, 'samples.csv'), index_col='id')
        self.ids = list(self.samples.index)
        print(self.samples.columns)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample_id = self.ids[item]
        image_path = os.path.join(self.path, self.samples['image'][sample_id])
        matte_path = os.path.join(self.path, self.samples['matte'][sample_id])

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image[:, :, :3] = image[:, :, :3][:, :, ::-1]
        matte = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)

        if self.resize is not None:
            image = cv2.resize(image, self.resize, interpolation=cv2.INTER_AREA)
            matte = cv2.resize(matte, self.resize, interpolation=cv2.INTER_AREA)

        return {
            'image': image, # (H, W, 4)
            'matte': matte, # (H, W)
        }


class MODNetMetaHumanHairMattingDataset(Dataset):
    """
    Implemented based on: https://github.com/ZHKKKe/MODNet/issues/200
    """
    def __init__(self, path='/datasets/generated/metahuman_matting_second_test/', resize=None):
        self.raw_dataset = MetaHumanHairMattingDataset(path, resize)
        self.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item):
        sample = self.raw_dataset[item]
        img = sample['image']
        mask = sample['matte']

        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # convert Image to pytorch tensor
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if self.transform:
            img = self.transform(img)
            trimap = self.get_trimap(mask)
            mask = self.transform(mask)

        img = torch.squeeze(img, 0)
        mask = torch.squeeze(mask, 0)[None]   # I added the none part because the trainer expects 4D tensor
        trimap = torch.squeeze(trimap, 1)
        return img, trimap, mask

    def get_trimap(self, alpha):
        # alpha \in [0, 1] should be taken into account
        # be careful when dealing with regions of alpha=0 and alpha=1
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        # image dilation implemented by Euclidean distance transform
        unknown = ndi.distance_transform_edt(unknown == 0) <= np.random.randint(1, 20)
        trimap = fg
        trimap[unknown] = 0.5
        return torch.unsqueeze(torch.from_numpy(trimap), dim=0)


if __name__ == '__main__':
    dataset = MetaHumanHairMattingDataset()
    sample = dataset[0]
    _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.imshow(sample['image'])
    ax2.imshow(sample['matte'])
    plt.show()
