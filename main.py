'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
'''

from __future__ import print_function

import os
import copy
import sys
import argparse
import shutil
import time
import math
import random
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from datasets import TinyImagenet
from fimportance import PeakyImportance
from saliency import get_param_importance
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, load_model
from networks.resnet_big import SupConResNet
from losses_negative_only import SupConLoss


# import apex
# from apex import amp, optimizers


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--target_task', type=int, default=0)

    parser.add_argument('--resume_target_task', type=int, default=None)

    parser.add_argument('--end_task', type=int, default=None)

    parser.add_argument('--replay_policy', type=str, choices=['random'], default='random')

    parser.add_argument('--mem_size', type=int, default=200)

    parser.add_argument('--cls_per_task', type=int, default=2)

    parser.add_argument('--distill_power', type=float, default=1.0)

    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')

    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tiny-imagenet', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--emb_dim', type=int, default=128)

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # LASP
    parser.add_argument('--distill_type', type=str, choices=['None', 'SD'], default='None')
    parser.add_argument('--reg_type', type=str, choices=['None', 'GM'], default='None')
    parser.add_argument('--onlypast', action='store_true',
                        help='compute parameter importance only based on samples stored in the memory')
    parser.add_argument('--onlycurrent', action='store_true',
                        help='compute parameter importance only based on samples from upcoming task')
    parser.add_argument('--selectiveness', type=float, default=1e-5,
                        help='parameter for feature importance modulate')

    opt = parser.parse_args()

    opt.save_freq = opt.epochs // 2

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        pass

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '/home/rmeshkin/Co2L/data'

    opt.variant = (opt.distill_type if opt.distill_type != 'None' else '') + \
                  (opt.reg_type if opt.reg_type != 'None' else '')
    if opt.variant == '':
        opt.variant = 'None'

    if opt.onlypast:
        ablation = 'onlypast'
    elif opt.onlycurrent:
        ablation = 'onlycurrent'
    else:
        ablation = 'combined'

    opt.model_path = './save_{}_{}_{}_{}/{}_models'.format(opt.variant, opt.replay_policy, opt.mem_size, ablation,
                                                           opt.dataset)
    opt.tb_path = './save_{}_{}_{}_{}/{}_tensorboard'.format(opt.variant, opt.replay_policy, opt.mem_size, ablation,
                                                             opt.dataset)
    opt.log_path = './save_{}_{}_{}_{}_{}/logs'.format(opt.variant, opt.replay_policy, opt.mem_size, ablation,
                                                       opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_{}_{}_{}_{}'. \
        format(opt.dataset, opt.variant, opt.size, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp,
               opt.trial,
               opt.start_epoch if opt.start_epoch is not None else opt.epochs, opt.epochs,
               opt.current_temp,
               opt.past_temp,
               opt.distill_power)

    if opt.variant != 'None':
        opt.model_name = '{}_{}'.format(opt.model_name, opt.selectiveness)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_replay_samples(opt, model, prev_indices=None):
    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # construct data loader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       transform=val_transform,
                                       download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                   transform=val_transform,
                                   download=True)
        val_targets = val_dataset.targets

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task * opt.cls_per_task))
    else:
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                prev_indices += torch.tensor(_prev_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(
            range(max(opt.target_task - 1, 0) * opt.cls_per_task, (opt.target_task) * opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices

    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)

    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = (
                    (opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float - ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (
                    len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        mask = val_targets[observed_indices] == c
        selected_observed_indices += torch.tensor(observed_indices)[mask][
            torch.randperm(mask.sum())[:size_for_c]].tolist()
    print(np.unique(val_targets[selected_observed_indices], return_counts=True))

    model.is_training = is_training

    return prev_indices + selected_observed_indices


def set_loader(opt, replay_indices):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset == 'tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                               p=0.5 if opt.size > 32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task))
    print(target_classes)

    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        for tc in target_classes:
            # target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset = Subset(_train_dataset, subset_indices)
        if len(replay_indices) > 0:
            replay_dataset = Subset(_train_dataset, replay_indices)
        else:
            replay_dataset = None
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        _train_dataset = TinyImagenet(root=opt.data_folder,
                                      transform=TwoCropTransform(train_transform),
                                      download=True)
        for tc in target_classes:
            target_class_indices = np.where(_train_dataset.targets == tc)[0]
            subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset = Subset(_train_dataset, subset_indices)
        if len(replay_indices) > 0:
            replay_dataset = Subset(_train_dataset, replay_indices)
        else:
            replay_dataset = None
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                             transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    if replay_dataset is not None:
        replay_loader = torch.utils.data.DataLoader(
            replay_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)
    else:
        replay_loader = None
    return train_loader, replay_loader, subset_indices


def set_model(opt):
    model = SupConResNet(name=opt.model, emb_dim=opt.emb_dim)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, model2, criterion, optimizer, epoch, select, param_salience, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()

    if (select > 0).sum() > 1 and opt.distill_type == 'SD':
        SD = True
    else:
        SD = False

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, encoded = model(images, return_feat=True)

        # IRD (current)
        if opt.target_task > 0:
            features1_prev_task = features
            if SD:
                sample_cnt = features.shape[0]
                features1_prev_task = features1_prev_task[select.repeat(sample_cnt, 1)].view(sample_cnt, -1)
                features1_prev_task = nn.functional.normalize(features1_prev_task, 2, 1)

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        # Asym SupCon
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        features_curr = features

        loss = criterion(features_curr, labels, target_labels=list(
            range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task)))

        # IRD (past)
        if opt.target_task > 0:
            with torch.no_grad():
                features2_prev_task = model2(images)
                if SD:
                    sample_cnt = features2_prev_task.shape[0]
                    features2_prev_task = features2_prev_task[select.repeat(sample_cnt, 1)].view(sample_cnt, -1)
                    features2_prev_task = nn.functional.normalize(features2_prev_task, 2, 1)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                logits_max2, _ = torch.max(features2_sim * logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                    features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            distill_loss = opt.distill_power * loss_distill
            distill.update(loss_distill.item(), bsz)
        else:
            distill_loss = 0

        # update metric
        loss += distill_loss
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()

        # gradient modulation
        if opt.reg_type == 'GM' and len(param_salience) > 0:
            for name, param in model.named_parameters():
                if name.split('.')[-1] == 'weight':
                    # plot_weights_grads_salience(name, param, self.param_salience[name])
                    param.grad *= 1 - param_salience[name].clamp(max=1)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx + 1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, distill=distill))
            sys.stdout.flush()

    return losses.avg, model2


def get_select_and_salience(train_loader, replay_loader, model, fi_module, opt, onlypast=False, onlycurrent=False):
    images, labels = next(iter(train_loader))

    # getting rid of replay samples in the first training batch
    mask = (opt.target_task * opt.cls_per_task <= labels) & (labels < (opt.target_task + 1) * opt.cls_per_task)
    current_labels = labels[mask].clone()
    current_images = torch.cat([images[0][mask].clone(), images[1][mask].clone()], dim=0)

    replay_images, replay_labels = next(iter(replay_loader))
    past_labels = replay_labels
    past_images = torch.cat([replay_images[0].clone(), replay_images[1].clone()], dim=0)

    images = torch.cat([images[0], images[1]], dim=0)

    if not onlypast and not onlycurrent:
        salience_images = torch.cat([current_images, past_images], dim=0)
        salience_labels = torch.cat([current_labels, current_labels, past_labels, past_labels], dim=0)
    elif onlycurrent:
        salience_images = current_images.clone()
        salience_labels = torch.cat([current_labels, current_labels], dim=0)
    else:
        salience_images = past_images.clone()
        salience_labels = torch.cat([past_labels, past_labels], dim=0)

    if torch.cuda.is_available():
        salience_images = salience_images.cuda(non_blocking=True)
        salience_labels = salience_labels.cuda(non_blocking=True)
    del images, labels, mask
    del current_images, current_labels
    del replay_images, replay_labels
    del past_images, past_labels
    del train_loader
    torch.cuda.empty_cache()

    if opt.distill_type != 'None' and opt.reg_type == 'None':
        with torch.no_grad():
            y = model(salience_images)
        select = fi_module(y, salience_labels).detach()[0:1]
        param_salience = []
    else:
        param_salience, select = get_param_importance(model, salience_images, salience_labels, fi_module)
        select = select[0:1]
    print('select non zero', select.shape, (select > 0).sum())
    return select, param_salience


def main():
    opt = parse_option()

    target_task = opt.target_task

    # build model and criterion
    model, criterion = set_model(opt)
    model2, _ = set_model(opt)
    model2.eval()

    # build optimizer
    optimizer = set_optimizer(opt, model)

    replay_indices = None

    fi_module = PeakyImportance(50, 300, opt.selectiveness)

    if opt.resume_target_task is not None:
        load_file = os.path.join(opt.save_folder,
                                 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy,
                                                                          target_task=opt.resume_target_task))
        model, optimizer = load_model(model, optimizer, load_file)
        if opt.resume_target_task == 0:
            replay_indices = []
        else:
            replay_indices = np.load(
                os.path.join(opt.log_folder,
                             'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy,
                                                                                target_task=opt.resume_target_task))
            ).tolist()
        print(len(replay_indices))


    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    original_epochs = opt.epochs

    if opt.end_task is not None:
        if opt.resume_target_task is not None:
            assert opt.end_task > opt.resume_target_task
        opt.end_task = min(opt.end_task + 1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task

    select = torch.full((opt.emb_dim,), False).cuda()
    param_salience = []
    model2 = None
    for target_task in range(0 if opt.resume_target_task is None else opt.resume_target_task + 1, opt.end_task):

        opt.target_task = target_task
        del model2
        torch.cuda.empty_cache()

        print('Start Training current task {}'.format(opt.target_task))

        # acquire replay sample indices
        replay_indices = set_replay_samples(opt, model, prev_indices=replay_indices)

        np.save(
            os.path.join(opt.log_folder, 'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy,
                                                                                            target_task=target_task)),
            np.array(replay_indices))

        # build data loader (dynamic: 0109)
        train_loader, replay_loader, subset_indices = set_loader(opt, replay_indices)

        np.save(
            os.path.join(opt.log_folder, 'subset_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy,
                                                                                            target_task=target_task)),
            np.array(subset_indices))

        if target_task > 0 and (opt.distill_type != 'None' or opt.reg_type != 'None'):
            select_task, param_salience = get_select_and_salience(copy.deepcopy(train_loader), replay_loader, model,
                                                                  fi_module, opt, opt.onlypast, opt.onlycurrent)
            select = select_task > 0
        else:
            select = torch.full((512,), True).cuda()

        model2 = copy.deepcopy(model)

        # training routine
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, model2 = train(train_loader, model, model2, criterion, optimizer, epoch, select, param_salience, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss_{target_task}'.format(target_task=target_task), loss, epoch)
            logger.log_value('learning_rate_{target_task}'.format(target_task=target_task),
                             optimizer.param_groups[0]['lr'], epoch)

        # save the last model
        save_file = os.path.join(
            opt.save_folder,
            'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy, target_task=target_task))
        save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
