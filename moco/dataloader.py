import os
import logging
from os.path import join as ojoin
from torch.utils.data import Dataset
from PIL import Image
import random
from moco.data_utils import (
    check_for_folder_structure,
    load_real_paths,
    load_syn_paths,
)


class ImageDataset(Dataset):
    def __init__(
        self,
        datadir,
        transform,
        transform_k=None,
        num_imgs=0,
        aug_type="default",
        epochs=0,
    ):
        """Initializes image paths and preprocessing module."""
        self.is_folder_struct = check_for_folder_structure(datadir)

        if aug_type == "disco" or aug_type == "disco_HF" or not self.is_folder_struct:
            self.img_paths = load_syn_paths(datadir, num_imgs)
        else:
            self.img_paths = load_real_paths(datadir, num_imgs)

        self.aug_disco = aug_type == "disco"
        self.aug_disco_hf = aug_type == "disco_HF"
        self.num_imgs = num_imgs

        self.epochs = epochs
        self.transform_q = transform
        self.transform_k = transform if transform_k is None else transform_k
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def disco_augmentation(self, index):
        person_path = self.img_paths[index]
        # simulate online augmentation
        imgs = sorted(os.listdir(person_path))[: self.epochs * 2]
        img_file1 = random.choice(imgs)
        img_file2 = random.choice(imgs)
        if self.aug_disco_hf:
            img_file1 = imgs[0]
            img_file2 = img_file1

        img1 = Image.open(ojoin(person_path, img_file1))
        img1 = img1.convert("RGB")
        img2 = Image.open(ojoin(person_path, img_file2))
        img2 = img2.convert("RGB")
        q = self.transform_q(img1)
        k = self.transform_k(img2)
        return [q, k], index

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        if self.aug_disco or self.aug_disco_hf:
            return self.disco_augmentation(index)
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        q = self.transform_q(image)
        k = self.transform_k(image)
        return [q, k], index

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


class InferenceDataset(Dataset):
    def __init__(self, datadir, transform, num_imgs=0, synthetic=True):
        """Initializes image paths and preprocessing module."""
        if synthetic:
            self.img_paths = load_syn_paths(datadir, num_imgs)
        else:
            self.img_paths = load_real_paths(datadir, num_imgs)

        self.transform = transform
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        img = self.transform(image)
        return img, self.img_paths[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def load_supervised_paths(datadir, num_persons):
    img_paths, labels = [], []
    id_folders = sorted(os.listdir(datadir))[:num_persons]
    for id in id_folders:
        id_path = ojoin(datadir, id)
        img_files = sorted(os.listdir(id_path))
        img_paths += [ojoin(id_path, f_name) for f_name in img_files]

        labels += [int(id)] * len(img_files)

    return img_paths, labels


class SupervisedDataset(Dataset):
    def __init__(self, datadir, transform, num_persons=0):
        """Similar to ImageDataset, but limit the number of persons"""
        self.img_paths, self.labels = load_supervised_paths(datadir, num_persons)
        self.transform = transform
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns with corresponding label."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        img = self.transform(image)
        return img, self.labels[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)
