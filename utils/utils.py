import numpy as np
from easydict import EasyDict as edict
import yaml


def count_top1_top5(labels, preds):
    assert len(labels) == len(preds), 'Inconsistent length between labels and predictions (%d vs %d)' % (len(labels), len(preds))
    top1 = 0.0
    top5 = 0.0
    for n in range(len(labels)):
        argsort = np.argsort(preds[n])
        if labels[n] in argsort[-1:]:
            top1 += 1
        if labels[n] in argsort[-5::]:
            top5 += 1
    return top1, top5


def config_from_file(cfg_file):
    with open(cfg_file, 'r') as fp:
        cfg = yaml.safe_load(fp)
    cfg = edict(cfg)
    return cfg
