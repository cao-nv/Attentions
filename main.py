# import numpy as np
from tensorboardX import SummaryWriter
from utils import CIFAR100, count_top1_top5, config_from_file
from models import create_model
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import ipdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./configs/config.yaml', help='configuration file')
    parser.add_argument('--arch', type=str,
                        default='resnet34', help='ResNet architecture')
    parser.add_argument('--attention', type=str,
                        default='None', help='Attention type: [None, SE, BAM, CAM....]')
    parser.add_argument('--resume', action='store_true',
                        help='resume')
    parser.add_argument('--test', action='store_true',
                        help='Test only')
    parser.add_argument('--load-epoch', type=int,
                        default=100, help='Epoch to load for testing')
    args = parser.parse_args()
    return args


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

        
def count_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def trainval(model, save_dir, cfg, resume_epoch=None):
    train_dataset = CIFAR100(data_dir, augmentation=cfg.TRAIN.AUGMENTATION)
    test_dataset = CIFAR100(data_dir, is_test=True, augmentation=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=40)
    
    cretirion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.DECAY_EPOCH, gamma=cfg.TRAIN.LR_DECAY_GAMMA)
    logdir = os.path.join(cfg.LOG_DIR, save_dir.split('/')[-1])
    summaryWriter = SummaryWriter(logdir)
    summaryWriter.add_graph(model, torch.zeros(1, 3, 32, 32))

    model.cuda()
    model = nn.DataParallel(model)
    start_epoch = -1
    iter_counter = 0
    if resume_epoch is not None:
        load_point = os.path.join(save_dir, 'model_{}.pth'.format(resume_epoch))
        saved_dict = torch.load(load_point)
        model.load_state_dict(saved_dict['state_dict'])
        start_epoch = saved_dict['epoch']
        iter_counter = saved_dict['iter_counter']
    for epoch in range(0, start_epoch+1):
        lrScheduler.step()
    print('Start at epoch %d' % (start_epoch+1))
    
    for epoch in range(start_epoch+1, cfg.TRAIN.EPOCH):
        for (images, labels) in tqdm(train_loader, desc='Epoch {}/{}'.format(epoch+1, cfg.TRAIN.EPOCH)):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = cretirion(outputs, labels)
            summaryWriter.add_scalar('train/loss', loss.cpu().item(), iter_counter)
            iter_counter += 1
            loss.backward()
            optimizer.step()
        lrScheduler.step()
        if epoch % cfg.VALID_STEP == (cfg.VALID_STEP - 1):
            model.eval()
            top1_acc, top5_acc = valid(model, test_dataset, cfg)
            model.train()
            summaryWriter.add_scalar('valid/top1', top1_acc, epoch+1)
            summaryWriter.add_scalar('valid/top5', top5_acc, epoch+1)
        
        if epoch % cfg.SAVE_STEP == (cfg.SAVE_STEP - 1):
            save_dict = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'iter_counter': iter_counter}
            filename = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
            torch.save(save_dict, filename)
    return model


def valid(model, dataset, cfg):
    top1_TP = 0.0
    top5_TP = 0.0
    loader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, num_workers=4)
    # with torch.no_grad():
    for i, (images, labels) in tqdm(enumerate(loader), desc='Test'):
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        top1, top5 = count_top1_top5(labels.cpu().numpy(), output.cpu().detach().numpy())
        top1_TP += top1
        top5_TP += top5
    top1_acc = top1_TP/len(dataset)
    top5_acc = top5_TP/len(dataset)
    return top1_acc, top5_acc


if __name__ == '__main__':
    args = parse_args()
    cfg = config_from_file(args.config)
    data_dir = cfg.DATA_DIR
    
    arch = args.arch
    attention = args.attention
    checkpoint = cfg.CHECKPOINT
    save_dir = os.path.join(checkpoint, '_'.join([arch, attention]))

    mkdir(checkpoint)
    mkdir(save_dir)

    model = create_model(arch, attention)
    num_params = count_params(model)
    print('{}_{}: {} parameters'.format(args.arch, args.attention, num_params))
    if args.resume:
        model = trainval(model, save_dir, cfg, resume_epoch=args.load_epoch)
    else:
        model = trainval(model, save_dir, cfg)
