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
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from dataset.dataset_test import CTDataset
from ops.dataset_ops import Train_Collatefn
from ops.dist_ops import synchronize, all_reduce
from ops.log_ops import setup_logger
from ops.stat_ops import ScalarContainer, count_consective_num
from ops.acc_ops import topks_correct, topk_errors, topk_accuracies
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DIST_FLAG = torch.cuda.device_count() > 1

local_rank = 0
CFG_FILE = "cfgs/test_bc.yaml"

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
                     target='train')

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
                                    shuffle=False)

Val_CE, Val_Acc = [ScalarContainer() for _ in range(2)]
logger.info("Do evaluation...")

gts, pcovs, yhats = [], [], []

with torch.no_grad():
    for i, (all_F, all_L, all_info) in enumerate(ValidLoader):

        labels = all_L.cuda()
        inputs = all_F.cuda()
        val_loss, val_acc, probs, yhat, A = model.calculate_objective([inputs], labels)  
        #val_loss = criterion(preds, labels)
        #val_acc = topk_accuracies(preds, labels, [1])[0]

        name = all_info[0]["name"]
        pid = name.split('/')[-3]
        #print(A.max(), A)

        #prob_preds = F.softmax(preds, dim=1)
        prob_ncov = probs[0].item()
        #prob_ncov = prob_preds[0, 1].item()
        gt = labels.item()
        
        gts.append(gt)
        pcovs.append(prob_ncov)
        yhats.append(yhat.item())
        print ("{} {} {} {}".format(all_info[0]["name"], pid, prob_ncov, labels.item()))
        Val_CE.write(val_loss); Val_Acc.write(val_acc)

from metrics import sensitivity_specificity
Ece, Eacc = Val_CE.read(), Val_Acc.read()
gts, yhats, pcovs = np.asarray(gts), np.asarray(yhats), np.asarray(pcovs)
ss, sc, Eauc = sensitivity_specificity(gts, pcovs)
kappa_score = cohen_kappa_score(gts, yhats)
cm = confusion_matrix(gts, yhats)
precision = precision_score(gts, yhats, average='macro')
recall = recall_score(gts, yhats, average='macro')
f1 = f1_score(gts, yhats, average='macro')
e = 0
logger.info("VALIDATION | E [{}] | CE: {:1.5f} | ValAcc: {:1.3f} | Valkappa: {:1.3f} | ValAuc: {:1.3f} | F1 score:{:1.3f} | Precision:{:1.3f} | Recall:{:1.3f}".format(e, Ece, Eacc, kappa_score, Eauc, f1, precision, recall))
logger.info(f"confusion matrix: {cm}")

#Draw ROC curve and confusion matrix
# false_positive = [(1-i) for i in sc]
# plt.figure()
# lw = 2
# plt.plot(false_positive, ss, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % Eauc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.savefig('roc_bc.jpg')


array = cm
df_cm = pd.DataFrame(cm, index = [i for i in ["Non-COVID-19", "COVID-19"]],
                  columns = [i for i in ["Non-COVID-19", "COVID-19"]])

f, ax = plt.subplots(figsize = (14, 10))
sn.set(font_scale=1.5)
sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d", annot_kws={"fontsize":60})
ax.tick_params(labelsize=35)
f.savefig('cm_bc.jpg')

# to_save = {'false_positive':false_positive, 'ss':ss}
# np.save('ours_bc_roc.npy', to_save) 