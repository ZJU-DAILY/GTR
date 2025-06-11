import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, accuracy_score, fowlkes_mallows_score, normalized_mutual_info_score, adjusted_mutual_info_score
import torch
from dtw import dtw
from scipy.spatial.distance import directed_hausdorff
import re
import pandas as pd
from tqdm import tqdm


def nmi_score(y, y_pred):
    return normalized_mutual_info_score(y, y_pred)


def ami_score(y, y_pred):
    return adjusted_mutual_info_score(y, y_pred)


def ari_score(y, y_pred):
    return adjusted_rand_score(y, y_pred)


def fms_score(y, y_pred):
    return fowlkes_mallows_score(y, y_pred)


def cluster_acc(y_true, y_pred):
    """
    Calculate unsupervised clustering accuracy. Requires scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity
    https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return accuracy_score(y_pred_voted, y_true)


def recall_at_k(predictions, true_ids, k):
    recall_scores = []
    for i in range(predictions.size(0)):
        top_k_preds = torch.topk(predictions[i], k).indices
        if true_ids[i] in top_k_preds:
            recall_scores.append(1)
        else:
            recall_scores.append(0)
    return np.mean(recall_scores)

