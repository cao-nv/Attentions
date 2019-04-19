import numpy as np
from tensorboardX import SummaryWriter
from utils import CIFAR100, count_top1_top5
from models import resnet34, resnet50, SE, BAM, CBAM
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import yaml
import ipdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./configs/config.yaml', help='configuration file')
    parser.add_argument('--arch', type=str,
                        default='resnet34', help='ResNet architecture')
    parser.add_argument('--attention', type=str,
                        default='None', help='Attention type: [None, SE, BAM, CAM]')
    # parser.add_argument('--resume', action='store_true',
    #                     help='resume')
    parser.add_argument('--test', action='store_true',
                        help='Test only')
    parser.add_argument('--load_epoch', type=int,
                        default=100, help='Epoch to load for testing')
    args = parser.parse_args()
    return args


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
            
def createModel(arch, attention, num_classes=100):
    if arch == 'resnet34':
        network = resnet34
    else:
        network = resnet50
    assert attention in ['None', 'SE', 'BAM', 'CBAM'], 'Wrong attention type: {}'.format(attention)
    if attention == 'SE':
        attention_dict = {'type': SE, 'reduce_rate': 16}
    elif attention == 'BAM':
        attention_dict = {'type': BAM, 'reduce_rate': 16}
    elif attention == 'CBAM':
        attention_dict = {'type': CBAM, 'reduce_rate': 16}
    else:
        attention_dict = None

    model = network(False, num_classes=100, attention_dict=attention_dict)
    return model


def trainval(model, save_dir, cfg):
    train_dataset = CIFAR100(data_dir, augmentation=cfg['TRAIN']['AUGMENTATION'])
    test_dataset = CIFAR100(data_dir, is_test=True, augmentation=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=20)
    
    cretirion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['TRAIN']['LR'], momentum=0.9, weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    # optimizer = optim.Adam(model.parameters(), lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg['TRAIN']['DECAY_EPOCH'], gamma=cfg['TRAIN']['LR_DECAY_GAMMA'])
    logdir = os.path.join(cfg['LOG_DIR'], save_dir.split('/')[-1])
    summaryWriter = SummaryWriter(logdir)
    summaryWriter.add_graph(model, torch.zeros(1, 3, 32, 32))
    
    model.cuda()
    model = nn.DataParallel(model)
    iter_counter = 0
    for epoch in range(cfg['TRAIN']['EPOCH']):
        for (images, labels) in tqdm(train_loader, desc='Epoch {}/{}'.format(epoch+1, cfg['TRAIN']['EPOCH'])):
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
        if epoch % cfg['VALID_STEP'] == (cfg['VALID_STEP'] - 1):
            model.eval()
            top1_acc, top5_acc = valid(model, test_dataset, cfg)
            model.train()
            summaryWriter.add_scalar('valid/top1', top1_acc, epoch+1)
            summaryWriter.add_scalar('valid/top5', top5_acc, epoch+1)
        
        # if (epoch+1) in cfg['TRAIN']['DECAY_EPOCH']:
        #     BATCH_SIZE = BATCH_SIZE * 2
        #     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)
        if epoch % cfg['SAVE_STEP'] == (cfg['SAVE_STEP'] - 1):
            save_dict = {'epoch': epoch,
                         'state_dict': model.state_dict()}
            filename = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
            torch.save(save_dict, filename)
    return model


def valid(model, dataset, cfg):
    top1_TP = 0.0
    top5_TP = 0.0
    loader = DataLoader(dataset, batch_size=cfg['TEST']['BATCH_SIZE'], num_workers=4)
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), desc='Test'):
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            top1, top5 = count_top1_top5(labels.cpu().numpy(), output.cpu().numpy())
            top1_TP += top1
            top5_TP += top5
    top1_acc = top1_TP/len(dataset)
    top5_acc = top5_TP/len(dataset)
    return top1_acc, top5_acc


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as fp:
        cfg = yaml.safe_load(fp)
    data_dir = cfg['DATA_DIR']
    
    arch = args.arch
    attention = args.attention
    checkpoint = cfg['CHECKPOINT']
    save_dir = os.path.join(checkpoint, '_'.join([arch, attention]))

    mkdir(checkpoint)
    mkdir(save_dir)

    model = createModel(arch, attention)
    # train_dataset = CIFAR100(data_dir, cfg['AUGMENTATION'])
    # test_dataset = CIFAR100(data_dir, is_test=True, augmentation=False)
    model = trainval(model, save_dir, cfg)
