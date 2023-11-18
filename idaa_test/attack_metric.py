#!/usr/bin/env python
# coding=utf-8
import numpy as np
from configure import *
import pdb

class AttackMetric:
    def __init__(self, src_labels, target_labels, white_pred_confidences, black_pred_confidences):
        self.src_labels = src_labels
        self.target_labels = target_labels
        self.white_pred_labels = np.argmax(white_pred_confidences, axis=1)
        self.black_pred_labels = np.argmax(black_pred_confidences, axis=1)
        self.white_error_set = self.src_labels != self.white_pred_labels
        self.white_targeted_success_set = self.white_pred_labels == self.target_labels
        self.white_pred_confidences = white_pred_confidences
        self.black_pred_confidences = black_pred_confidences

    def error_rate(self):
        return np.sum(self.src_labels != self.black_pred_labels) / len(self.black_pred_labels)

    def targeted_success_rate(self):
        return np.sum(self.black_pred_labels == self.target_labels) / len(self.black_pred_labels)
