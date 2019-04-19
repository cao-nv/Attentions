import numpy as np


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
