import os
from os.path import join as ojoin


def check_for_folder_structure(datadir):
    """checks if datadir contains folders (like CASIA) or images (synthetic datasets)"""
    img_path = sorted(os.listdir(datadir))[0]
    img_path = ojoin(datadir, img_path)
    return os.path.isdir(img_path)


def load_first_dfg_path(path, num_imgs):
    """loads complete image path of first image for each DFG class
    args:
        path: path to class folders
        num_imgs: number of images == number of classes
    return:
        list of image paths
    """
    img_paths = []
    id_folders = sorted(os.listdir(path))
    if num_imgs != 0:
        id_folders = id_folders[:num_imgs]
    for id in id_folders:
        img = sorted(os.listdir(ojoin(path, id)))[0]
        img_paths.append(ojoin(path, id, img))
    return img_paths


def load_real_paths(datadir, num_imgs=0):
    """loads complete real image paths
    args:
        datadir: path to image folders
        num_imgs: number of images
    return:
        list of image paths
    """
    img_paths = []
    id_folders = sorted(os.listdir(datadir))
    for id in id_folders:
        img_files = sorted(os.listdir(ojoin(datadir, id)))
        img_paths += [os.path.join(datadir, id, f_name) for f_name in img_files]
    if num_imgs != 0:
        img_paths = img_paths[:num_imgs]
    return img_paths


def load_syn_paths(datadir, num_imgs=0):
    """loads synthetic paths
    args:
        datadir: path to folder
        num_imgs: number of images
    return:
        list of image paths
    """
    img_files = sorted(os.listdir(datadir))
    if num_imgs != 0:
        img_files = img_files[:num_imgs]
    return [os.path.join(datadir, f_name) for f_name in img_files]
