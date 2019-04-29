import numpy as np
import torch
from models import create_model
from utils import CIFAR100, config_from_file
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import argparse
import os

import ipdb
RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD = (0.2023, 0.1994, 0.2010)
inv_normalize = transforms.Normalize(mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], std=[1/0.2023, 1/0.1994, 1/0.2010])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./configs/config.yaml', help='configuration file')
    parser.add_argument('--arch', type=str,
                        default='resnet50', help='ResNet architecture')
    parser.add_argument('--attention', type=str,
                        default='None', help='Attention type: [None, SE, BAM, CAM]')
    parser.add_argument('--load-epoch', type=int,
                        default=149, help='Epoch to load for testing')
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
    return args


def dict_nnDataParallel(state_dict):
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return new_state_dict


def get_gradCAM_heatmap(img_batch_1, model, label):
    pred = model(img_batch_1)
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred[:, label].backward()
    pred_prob = pred[:, label].item()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
    activations = model.get_activations(img_batch_1).detach()
    activations = activations * pooled_gradients
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap.data.cpu().numpy(), pred_prob


def view_gradCAM(img_batch_1, heatmap=None):
    img_batch_1 = img_batch_1.squeeze(dim=0)
    inv_norm_img = inv_normalize(img_batch_1)
    inv_norm_img_np = inv_norm_img.data.cpu().numpy()
    rgb_img = inv_norm_img_np.transpose((1, 2, 0))
    rgb_img = np.uint8(255*rgb_img)
    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = 0.4*heatmap + rgb_img
    else:
        superimposed_img = rgb_img
    return superimposed_img


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

        
if __name__ == '__main__':
    args = parse_args()
    cfg = config_from_file(args.config)
    data_dir = cfg.DATA_DIR
    arch = args.arch
    attention = args.attention
    output_dir = args.output_dir
    mkdir(output_dir)
    checkpoint = cfg.CHECKPOINT
    save_dir = os.path.join(checkpoint, '_'.join([arch, attention]))
    test_dataset = CIFAR100(data_dir, is_test=True, augmentation=False)
    if arch != 'None':
        model = create_model(arch, attention)
        load_point = os.path.join(save_dir, 'model_{}.pth'.format(args.load_epoch))
        saved_dict = torch.load(load_point)
        state_dict = dict_nnDataParallel(saved_dict['state_dict'])
        model.load_state_dict(state_dict)
        model.eval()
        for i in range(8):
            img, label = test_dataset[i]
            img = torch.unsqueeze(img, 0)
            heatmap, pred = get_gradCAM_heatmap(img, model, label)
            sp_img = view_gradCAM(img, heatmap)
            output_file = os.path.join(output_dir, '{}_{}_{}_{:.5f}.png'.format(i, arch, attention, pred))
            cv2.imwrite(output_file, sp_img)
    else:
        for i in range(8):
            img, label = test_dataset[i]
            sp_img = view_gradCAM(img, heatmap=None)
            fine_label = test_dataset.fine_label_names[label]
            output_file = os.path.join(output_dir, '{}_{}.png'.format(i, fine_label))
            cv2.imwrite(output_file, sp_img)
    print(output_dir)

    
