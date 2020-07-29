#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019-12-04 19:00 qiang.zhou <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

from importlib import import_module
import random, sys, yaml, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

from dataset.dataset_test import CTDataset
from ops.dataset_ops import Train_Collatefn
from ops.dist_ops import synchronize, all_reduce
from ops.log_ops import setup_logger
from ops.stat_ops import ScalarContainer, count_consective_num
from ops.acc_ops import topks_correct, topk_errors, topk_accuracies
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DIST_FLAG = torch.cuda.device_count() > 1

local_rank = 0
CFG_FILE = "cfgs/test_mc.yaml"

#fold_id = int(model_path.split('/')[-1][:2])

############### Set up Variables ###############
with open(CFG_FILE, "r") as f: cfg = yaml.safe_load(f)

DATA_ROOT = cfg["DATASETS"]["TEST_DATA_ROOT"]
MODEL_UID = cfg["MODEL"]["MODEL_UID"]
ARCH = cfg["MODEL"]["ARCH"]
DEPTH = cfg["MODEL"]["DEPTH"]
PRETRAINED_MODEL_PATH = cfg["MODEL"]["PRETRAINED_MODEL_PATH"]
NUM_CLASSES = cfg["MODEL"]["NUM_CLASSES"]
SAMPLE_NUMBER = int(cfg["DATALOADER"]["SAMPLE_NUMBER"])
NUM_WORKERS = int(cfg["DATALOADER"]["NUM_WORKERS"])
RESULE_HOME = cfg["TEST"]["RESULE_HOME"]
LOG_FILE = cfg["TEST"]["LOG_FILE"]

model = import_module(f"model.{MODEL_UID}")
ENModel = getattr(model, "ENModel")
criterion = torch.nn.CrossEntropyLoss(reduction="mean")

############### Set up Dataloaders ###############
Validset = CTDataset(datalist=DATA_ROOT,
                     target='valid',)

model = ENModel(arch=ARCH, resnet_depth=DEPTH, 
                input_channel=1, num_classes=NUM_CLASSES)
model = model.cuda()
############### Logging out some training info ###############
Epoch_CE, Epoch_Acc = [ScalarContainer() for _ in range(2)]
logger = setup_logger()

logger.info("Config {}...".format(CFG_FILE))
logger.info("{}".format(json.dumps(cfg, indent=1)))
logger.warning(f"Loading init model path {PRETRAINED_MODEL_PATH}")
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

ValidLoader = torch.utils.data.DataLoader(Validset,
                                    batch_size=1, 
                                    num_workers=NUM_WORKERS,
                                    collate_fn=Train_Collatefn,
                                    shuffle=False,)

Val_CE, Val_Acc = [ScalarContainer() for _ in range(2)]
logger.info("Do evaluation...")

gts, y_hat, probs = [], [], []

with torch.no_grad():
    for i, (all_F, all_L, all_info) in enumerate(ValidLoader):

        labels = all_L.cuda()
        preds, w = model([all_F.cuda()])

        val_loss = criterion(preds, labels)
        val_acc = topk_accuracies(preds, labels, [1])[0]

        name = all_info[0]["name"]
        pid = name.split('/')[-3]
        print(w.max())
        prob_preds = F.softmax(preds, dim=1)
        #prob_normal = prob_preds[0, 0].item()
        #prob_cp = prob_preds[0, 1].item()
        #prob_ncov = prob_preds[0, 2].item()
        pred_label = torch.argmax(prob_preds, dim=1).item()
        gt = labels.item()
        #print(prob_preds[0])
        gts.append(gt)
        y_hat.append(pred_label)
        probs.append(prob_preds[0].cpu().detach().numpy())

        print ("{} {} {} {}".format(all_info[0]["name"], pid, pred_label, labels.item()))

        Val_CE.write(val_loss); Val_Acc.write(val_acc)

from metrics import sensitivity_specificity
Ece, Eacc = Val_CE.read(), Val_Acc.read()
gts, y_hat, probs = np.asarray(gts), np.asarray(y_hat), np.asarray(probs)
kappa_score = cohen_kappa_score(gts, y_hat)
cm = confusion_matrix(gts, y_hat)
precision = precision_score(gts, y_hat, average='macro')
recall = recall_score(gts, y_hat, average='macro')
f1 = f1_score(gts, y_hat, average='macro')
auc = roc_auc_score(gts, probs, multi_class='ovr')
#_, _, Eauc = sensitivity_specificity(gts, pcovs)
e = 0
logger.info("VALIDATION | E [{}] | CE: {:1.5f} | ValAcc: {:1.3f} | Valkappa: {:1.3f} | ValAuc: {:1.3f} | F1 score:{:1.3f} | Precision:{:1.3f} | Recall:{:1.3f}".format(e, Ece, Eacc, kappa_score, auc, f1, precision, recall))

logger.info(f"confusion matrix: {cm}")

#TODO: Visulize t-sne, key instance, training/val loss curce, training/val accuracy curve. 
array = cm
df_cm = pd.DataFrame(cm, index = [i for i in ["NP", "CP", "COVID-19"]],
                  columns = [i for i in ["NP", "CP", "COVID-19"]])

f, ax = plt.subplots(figsize = (14, 10))
sn.set(font_scale=1.5)
sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d", annot_kws={"fontsize":40})
ax.tick_params(labelsize=30)
f.savefig('cm_ours_mc.jpg')