#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from torch import nn
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import save_one_img
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pickle
from attacks import idaa 
from attacks.tools.seed import random_seed

def main():
    mzoo = ModelZoo()
    dzoo = DatasetZoo()
    with open(attack_book, 'r') as fp:
        attack_targets = json.load(fp)

    for mname in model_names:
        print('model {} generates adversarial examples...'.format(mname))

        adv_output_dir = os.path.join(test_output_path, mname)
        if not os.path.exists(adv_output_dir):
            os.mkdir(adv_output_dir)

        for (dname, dpath) in victim_datasets:

            adv_output_dir = os.path.join(test_output_path, mname, dname)
            if not os.path.exists(adv_output_dir):
                os.mkdir(adv_output_dir)

            print('1. dataset {} is attacked...'.format(dname)) 
            ds = dzoo.load_dataset(dname, dpath)
            label_space = list(ds.class_to_idx.values())

            model = mzoo.pick_model(mname)
            feature_model, decision_model = mzoo.default_split(mname)
            model = model.cuda()
            model.eval()
            if feature_model is not None:
                feature_model = feature_model.cuda()
                decision_model = decision_model.cuda()
                feature_model.eval()
                decision_model.eval()

            for i, (attack_name, attack_args) in enumerate(attack_methods.items()):
                random_seed()

                adv_output_dir = os.path.join(test_output_path, mname, dname, attack_name)
                if not os.path.exists(adv_output_dir):
                    os.mkdir(adv_output_dir)
    
                print('2.{} attack method {} is attacking...'.format(i, attack_name))
                if attack_name == 'IDAA':
                    attack = idaa.IDAA(model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=attack_args['alpha'],
                            n=10, mixup_num=3, mixup_weight=0.4, mixup_ratio=0.7)
                    # targeted
                    attack.set_mode_targeted_by_label()
                else:
                    # raise 'Invalid attack method!!!'
                    continue

                # begin to attack
                adv_confidences = {} 
                for (feature, label), (fname, _) in tqdm(zip(ds, ds.imgs)):
                    feature = feature.unsqueeze(0).cuda()
                    source = torch.LongTensor([label]).cuda()

                    fname_basename = os.path.basename(fname)
                    (_, target) = attack_targets[fname_basename]
                    target = torch.LongTensor([target]).cuda()
                    adv_output_file = os.path.join(adv_output_dir, fname_basename)

                    adv_feature = attack(feature, target, source_labels=source) 
                    save_one_img(adv_feature.detach().cpu(), adv_output_file)

                    adv_confidence = F.softmax(model(adv_feature), dim=1)
                    adv_confidences[fname_basename] = adv_confidence.detach().cpu().numpy()

                adv_output_confidence = os.path.join(adv_output_dir, 'confidence.npy')
                with open(adv_output_confidence, 'wb') as fp:
                    pickle.dump(adv_confidences, fp)

if __name__ == '__main__':
    main()
