import os
import PIL.Image as Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, data_list):
    images = []
    for img_name, idx, labels in data_list:
        item = (img_name, int(idx))
        images.append(item)
    return images


class BirdImageLoader(Dataset):
    def __init__(
        self,
        root,
        data_list,
        class_to_idx,
        transform=None,
        target_transform=None,
    ):
        imgs = make_dataset(root, data_list)

        self.root = root
        self.imgs = imgs
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

class TestImageLoader(Dataset):
    def __init__(
        self,
        root,
        transform=None,
    ):
        
        imgs = []
        for img_name in os.listdir(root):
            if is_image_file(img_name):
                imgs.append(img_name)
        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        avg_R = np.average(img[0])
        avg_G = np.average(img[1])
        avg_B = np.average(img[2])
        if (avg_G - 0.05 > avg_B and avg_G - 0.05 > avg_R):
            img[1] = img[1] * 0.75
        if self.transform is not None:
            img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.imgs)
