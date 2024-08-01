import os
import json
import random
import torch
import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter

from ignite.metrics.metric import Metric


class SamplewiseAccuracy(Metric):
    """ Segmentation samplewise accuracy. This metric can be attached to 
    an ignite evaluator engine and will return the samplewise accuracy
    for each epoch."""

    def reset(self):
        """ Resets the number of correctly predicted and total samples
        at the start of each epoch. """
        self._correct_samples = 0
        self._total_samples = 0

    def update(self, outputs, labels):
        # Unpack data, assert shapes and get predictions
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update numbers of correctly predicted and total samples
        self._correct_samples += (outputs == labels).sum(dtype=torch.float)
        self._total_samples += torch.numel(outputs)

    def compute(self):
        return self._correct_samples / self._total_samples


class MeanAccuracy(Metric):
    """ Segmentation mean class accuracy. This metric can be attached to 
    an ignite evaluator engine and will return the mean class accuracy
    for each epoch."""

    def reset(self):
        """ Resets the classwise number of correctly predicted and total samples 
        at the start of each epoch. """
        self._correct_class_samples = {}
        self._total_class_samples = {}

    def update(self, outputs, labels):
        # Unpack data, assert shapes and get predictions
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update correctly predicted and total precitions for each class in batch
        for label in torch.unique(labels):
            if not label in self._total_class_samples:
                self._correct_class_samples[label] = 0
                self._total_class_samples[label] = 0

            # Samples belonging to current class
            class_samples = labels == label

            # Correctly predicted samples and total samples for current class in batch
            correct_samples = (outputs[class_samples] == label).sum(dtype=torch.float)
            total_samples = class_samples.sum(dtype=torch.float)
            self._correct_class_samples[label] += correct_samples
            self._total_class_samples[label] += total_samples

    def compute(self):
        accuracies = []
        for label in self._total_class_samples:
            correct_samples = self._correct_class_samples[label]
            total_samples = self._total_class_samples[label]
            accuracies.append(correct_samples / total_samples)
        return torch.mean(torch.tensor(accuracies))
    
    def compute_per_class(self):
        accuracies = []
        for label in self._total_class_samples:
            correct_samples = self._correct_class_samples[label]
            total_samples = self._total_class_samples[label]
            accuracies.append(correct_samples / total_samples)
        return torch.tensor(accuracies)


class MeanIoU(Metric):
    """ Segmentation mean class IoU. This metric can be attached to 
    an ignite evaluator engine and will return the mean IoU for each epoch."""

    def reset(self):
        """ Resets the classwise intersection and union at the start of each epoch."""
        self._class_intersection = {}
        self._class_union = {}

    def update(self, outputs, labels):
        # Unpack data, assert shapes and get predictions
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update intersection and union for each class in batch
        for label in torch.unique(labels):
            if not label in self._class_intersection:
                self._class_intersection[label] = 0
                self._class_union[label] = 0

            # Intersection and union of current class
            intersection = (
                ((labels == label) & (outputs == label)).sum(dtype=torch.float).item()
            )
            union = (
                ((labels == label) | (outputs == label)).sum(dtype=torch.float).item()
            )
            self._class_intersection[label] += intersection
            self._class_union[label] += union

    def compute(self):
        ious = []
        for label in self._class_intersection:
            total_intersection = self._class_intersection[label]
            total_union = self._class_union[label]
            ious.append(total_intersection / total_union)
        return torch.mean(torch.tensor(ious))


class FrequencyWeightedIoU(Metric):
    """ Segmentation frequency weighted class IoU. This metric can be attached to 
    an ignite evaluator engine and will return the frequency weighted IoU for each epoch."""

    def reset(self):
        """ Resets the classwise intersection, union, class samples and total samples at the start of each epoch."""
        self._class_intersection = {}
        self._class_union = {}
        self._class_samples = {}
        self._total_samples = 0

    def update(self, outputs, labels):
        # Unpack data, assert shapes and get predictions
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update intersection, union, class and total samples
        for label in torch.unique(labels):
            if not label in self._class_intersection:
                self._class_intersection[label] = 0
                self._class_union[label] = 0
                self._class_samples[label] = 0

            # Samples belonging to current class
            class_samples = labels == label

            # Total samples, class samples, and intersection and union of current class
            self._total_samples += class_samples.sum(dtype=torch.float).item()
            self._class_samples[label] += class_samples.sum(dtype=torch.float).item()
            intersection = (
                ((labels == label) & (outputs == label)).sum(dtype=torch.float).item()
            )
            union = (
                ((labels == label) | (outputs == label)).sum(dtype=torch.float).item()
            )
            self._class_intersection[label] += intersection
            self._class_union[label] += union

    def compute(self):
        ious = []
        for label in self._class_intersection:
            total_samples = self._total_samples
            class_samples = self._class_samples[label]
            class_intersection = self._class_intersection[label]
            class_union = self._class_union[label]
            ious.append(class_samples * class_intersection / class_union)
        return torch.tensor(ious).sum().item() / total_samples

    def computer_per_class(self):
        ious = []
        for label in self._class_intersection:
            class_samples = self._class_samples[label]
            class_intersection = self._class_intersection[label]
            class_union = self._class_union[label]
            ious.append(class_samples * class_intersection / class_union)
        return torch.tensor(ious)




def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mode_filter(x, size):
    def modal(P):
        mode = stats.mode(P)
        return mode.mode[0]
    result = generic_filter(x, modal, size)
    return result

############# Modified from ASFormer/MSTCN #################

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float32)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score

 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 
def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
 

def func_eval(label_dir, pred_dir, video_list):
    
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0

    for vid in video_list:
 
        gt_file = os.path.join(label_dir, f'{vid}.txt')
        gt_content = read_file(gt_file).split('\n')[0:-1]
 
        pred_file = os.path.join(pred_dir, f'{vid}.txt')
        pred_content = read_file(pred_file).split('\n')[1].split()
 
        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        edit += edit_score(pred_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(video_list)
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
 
    return acc, edit, f1s


############# Visualization #################

def plot_barcode(class_num, gt=None, pred=None, show=True, save_file=None):

    # if class_num <= 10:
    #     color_map = plt.cm.tab10
    # elif class_num > 20:
    #     color_map = plt.cm.gist_ncar
    # else:
    #     color_map = plt.cm.tab20

    # color_map = {}
    # for i in range(class_num):
    #     color_map[str(i)] = plt.cm.Set1(i)

    color_map = plt.cm.Set1

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, 
                interpolation='nearest', vmin=0, vmax=9)

    fig = plt.figure(figsize=(18, 18))

    # a horizontal barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.45, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        # ax1.imshow(gt.reshape((1, -1)), **barprops)
        ax1.imshow(gt, **barprops)

    if pred is not None:
        ax2 = fig.add_axes([0, 0.15, 1, 0.2], **axprops)
        ax2.set_title('Predicted')
        # ax2.imshow(pred.reshape((1, -1)), **barprops)
        ax2.imshow(pred, **barprops)

    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":

    gt = np.random.randint(5, size=(10,20))
    pred = np.random.randint(5, size=(10,20))

    plot_barcode(4, gt, pred, show=True, save_file='test.png')
