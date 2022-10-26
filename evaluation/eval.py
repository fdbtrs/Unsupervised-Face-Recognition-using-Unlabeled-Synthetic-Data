import logging
import argparse
import os
import torch
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging
from backbones.iresnet import iresnet50
from moco.builder import MoCo


def get_MoCo_backbone(embedding_size, model_path, device):
    """loads MoCo model encoder q
    args:
        embedding_size: embedding dimension
        model_path: path to model
        device: torch device
    return:
        model encoder
    """
    print("loading MoCo model...")
    backbone = MoCo(base_encoder=iresnet50, dim=embedding_size, K=32768).to(device)
    ckpt = torch.load(model_path, map_location=device)
    backbone.load_state_dict(ckpt, strict=False)
    return backbone.encoder_q


def get_resnet_backbone(embedding_size, model_path, device):
    """loads ResNet 50 model
    args:
        embedding_size: embedding dimension
        model_path: path to model
        device: torch device
    return:
        ResNet50
    """
    print("loading ResNet model...")
    backbone = iresnet50(num_features=embedding_size, use_se=False).to(device)
    ckpt = torch.load(model_path, map_location=device)
    backbone.load_state_dict(ckpt)
    return backbone


def get_backbone(architecture, model_path, embedding_size, device):
    """loads model encoder
    args:
        architecture: moco, resnet, simclr
        model_path: path to model
        embedding_size: embedding dimension
        device: torch device
    return:
        encoder
    """
    if architecture.lower() == "moco":
        backbone = get_MoCo_backbone(embedding_size, model_path, device)
    elif architecture.lower() == "resnet":
        backbone = get_resnet_backbone(embedding_size, model_path, device)
    else:
        print("Unknown Architecture:", architecture)
        exit()
    return torch.nn.DataParallel(backbone, device_ids=[device])


def main(args):
    val_targets = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]
    gen_im_path = None
    if args.evalset != "all":
        val_targets = [str(args.evalset)]
        gen_im_path = os.path.join(args.model_folder, val_targets[0])
    gpu_id = 0
    device = torch.device(gpu_id)

    log_root = logging.getLogger()
    init_logging(log_root, 0, args.model_folder, logfile="test.log")
    callback_verification = CallBackVerification(
        1, 0, val_targets, args.rec_path, (112, 112), gen_im_path=gen_im_path
    )

    if args.epoch > -1:
        w = "checkpoint_{:03d}.pth".format(args.epoch)
        model = get_backbone(
            args.architecture, os.path.join(args.model_folder, w), args.emb_dim, device
        )
        print("Loaded Model from", os.path.join(args.model_folder, w))
        callback_verification(args.epoch, model)

    else:
        weights = sorted(os.listdir(args.model_folder))
        for w in weights:
            if "checkpoint" in w:
                model = get_backbone(
                    args.architecture,
                    os.path.join(args.model_folder, w),
                    args.emb_dim,
                    device,
                )
                epoch = w.split("_")[-1].split(".")[0]
                callback_verification(int(epoch), model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ArcFace Training")
    parser.add_argument(
        "--model_folder",
        type=str,
        default="../output/RandAug_hyperparameter_exp/syn100K_RandAug2_24",
        help="model folder",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=40,
        help="which epoch should be evaluated; -1 eval all epochs",
    )
    parser.add_argument("--emb_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument(
        "--rec_path",
        type=str,
        default="/data/fboutros/faces_emore",
        help="Path to folder that includes the bin files",
    )
    parser.add_argument("--architecture", type=str, default="moco", help="moco, resnet")
    parser.add_argument(
        "--evalset",
        type=str,
        default="all",
        help="used for EER evaluation: all, lfw, cfp_fp, cfp_ff, agedb_30, calfw, cplfw",
    )

    args_ = parser.parse_args()
    main(args_)
