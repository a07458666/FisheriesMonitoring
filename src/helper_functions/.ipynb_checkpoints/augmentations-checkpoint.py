from PIL import ImageFilter
from torchvision import transforms
import random

imageSize = 224
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_all_in_aug(img_size=imageSize):
    all_in_transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=[0.5, 1.5]),
            transforms.RandomRotation(degrees=15),
            transforms.CenterCrop((img_size, img_size)),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return all_in_transform


def get_aug_trnsform(img_size=imageSize):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        0.4, 0.4, 0.4, 0.1
                    )  # not strengthened
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform


def get_eval_trnsform(img_size=imageSize):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform
