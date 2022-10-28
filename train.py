import argparse
import math
import os
import logging
import torch
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

import config.config as cfg
from moco.builder import MoCo
from moco.dataloader import ImageDataset
import utils.augmentation as augment
from utils.utils_logging import init_logging, AverageMeter
from utils.utils_callbacks import (
    CallBackLogging,
    CallBackVerification,
    CallBackModelCheckpoint,
)
from backbones.iresnet import iresnet100, iresnet50


def main(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cudnn.benchmark = True

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root, local_rank, cfg.output_dir, logfile="Training.log")

    ###############################################
    ####### Create Model + resume Training ########
    ###############################################
    logging.info("=> creating model '{}'".format(cfg.architecture))
    model = MoCo(
        base_encoder=iresnet50,
        dim=cfg.moco_dim,
        K=cfg.moco_k,
        m=cfg.moco_m,
        T=cfg.moco_t,
        mlp=cfg.mlp,
        queue_type=cfg.queue_type,
        margin=cfg.loss_margin,
        dropout=cfg.drop_ratio,
    ).to(local_rank)
    # print(model)
    start_epoch = 0
    if args.resume:
        try:
            backbone_pth = os.path.join(
                cfg.output_dir, "checkpoint_{:03d}.pth".format(cfg.resume_epoch)
            )
            model.load_state_dict(
                torch.load(backbone_pth, map_location=torch.device(local_rank))
            )
            start_epoch = cfg.resume_epoch

            if rank == 0:
                logging.info("backbone resume loaded successfully!")
                logging.info("resume from epoch {}".format(start_epoch))
                logging.info("remaining epochs {}".format(cfg.epochs - start_epoch))
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

    for ps in model.parameters():
        dist.broadcast(ps, 0)

    model = DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[local_rank]
    )
    model.train()

    ###############################################
    ######### loss function + optimizer ###########
    ###############################################
    criterion = CrossEntropyLoss().cuda(local_rank)

    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.learning_rate / 512 * cfg.batch_size * world_size,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10, cooldown=10
    )

    ###############################################
    ################ Data Loading #################
    ###############################################
    if cfg.augmentation == "h_flip":
        augmentation = transforms.Compose(augment.aug_h_flip)
    elif cfg.augmentation == "num_mag_exp":
        augmentation = transforms.Compose(augment.get_randaug(args.num_ops, args.mag))
    elif cfg.augmentation == "randaug_4_16":
        augmentation = transforms.Compose(augment.aug_rand_4_16)
    elif cfg.augmentation == "aug_operation_exp":
        logging.info("Augmentation under testing: " + args.aug_operation)
        augmentation = transforms.Compose(
            augment.select_x_operation(args.aug_operation)
        )
    elif cfg.augmentation == "disco":
        aug = augment.aug_rand_4_16
        augmentation = transforms.Compose(aug)
    elif cfg.augmentation == "disco_HF":
        augmentation = transforms.Compose(augment.aug_rand_4_16)
    else:
        logging.error("Unknown augmentation method: {}".format(cfg.augmentation))
        exit()

    train_dataset = ImageDataset(
        cfg.datapath,
        augmentation,
        num_imgs=cfg.number_of_images,
        aug_type=cfg.augmentation,
        epochs=cfg.epochs,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    ###############################################
    ################# Callbacks ###################
    ###############################################
    total_step = int(len(train_dataset) / cfg.batch_size / world_size * cfg.epochs)
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)
    print_freq = len(train_dataset) / cfg.batch_size / world_size / 10
    print_freq = max(round(print_freq / 10) * 10, 10)
    callback_logging = CallBackLogging(
        print_freq, rank, total_step, cfg.batch_size, world_size
    )
    callback_verification = CallBackVerification(
        1, rank, cfg.val_targets, cfg.eval_datasets, img_size=[112, 112]
    )
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output_dir)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global_step = 0
    e_acc = torch.zeros(2).to(rank)  # (epoch, maxAcc)
    avg_acc = torch.zeros(1).to(rank)

    ###############################################
    ################## Training ###################
    ###############################################
    for epoch in range(start_epoch, cfg.epochs):
        train_sampler.set_epoch(epoch)
        if not cfg.auto_schedule:
            adjust_learning_rate(optimizer, epoch, cfg)
        logging.info(f"learning rate: {round(optimizer.param_groups[0]['lr'], 12)}")

        # train for one epoch
        for i, (images, _) in enumerate(train_loader):

            images[0] = images[0].cuda(local_rank, non_blocking=True)
            images[1] = images[1].cuda(local_rank, non_blocking=True)

            # compute output
            output, target = model(im_q=images[0], im_k=images[1], epoch=epoch)

            loss = criterion(output, target)

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))
            top5.update(acc5[0], images[0].size(0))

            # compute gradient and do SGD step
            clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            callback_logging(global_step, losses, top1, top5, epoch)
            global_step += 1

        ver_accs = callback_verification(epoch + 1, model.module.encoder_q)
        avg_acc[0] = sum(ver_accs[:3]) / 3
        dist.broadcast(avg_acc, src=0)
        if cfg.auto_schedule:
            scheduler.step(avg_acc)
        # update max accuracy
        if rank == 0 and avg_acc[0] > e_acc[1]:
            e_acc[0] = epoch
            e_acc[1] = avg_acc[0]
            # only save models after 10 epochs to save disk space
            if cfg.auto_schedule and epoch >= 10:
                callback_checkpoint(epoch, model)
        dist.broadcast(e_acc, src=0)
        # save only last 2 epochs
        if epoch >= cfg.epochs - 2:
            callback_checkpoint(epoch, model)

        # early stopping
        if cfg.auto_schedule and e_acc[0] <= epoch - 20:
            callback_checkpoint(epoch, model)
            logging.info(
                "Avg validation accuracy did not improve for 20 epochs. Terminating..."
            )
            exit()


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.learning_rate
    if cfg.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / cfg.epochs))
    else:  # stepwise lr schedule
        for milestone in cfg.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = (
                correct[:k].reshape(k * correct.shape[1]).float().sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Master MoCo Training")
    parser.add_argument("--resume", type=int, default=0, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    args = parser.parse_args()
    main(args)
