import torchvision.transforms as transforms
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.rand_augment import RandAugment


normalize_moco = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
aug_default = [
    transforms.RandomResizedCrop(112, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_moco,
]

to_tensor = [transforms.ToTensor(), normalize]

only_normalize = [normalize]

aug_h_flip = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]

aug_rand_4_16 = [
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=4, magnitude=16),
    transforms.ToTensor(),
    normalize,
]


def get_randaug(n, m):
    """return RandAugment transforms with
    n: number of operations
    m: magnitude
    """
    return [
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=n, magnitude=m),
        transforms.ToTensor(),
        normalize,
    ]


def select_x_operation(x):
    """enable only the x operation to RandAug
    x: string of the available augmentation
    """
    return [
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=1, magnitude=9, available_aug=x),
        transforms.ToTensor(),
        normalize,
    ]
