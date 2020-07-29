from importlib import import_module
import random, sys, yaml, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
import logging
from dataset.dataset_test import CTDataset
from ops.dataset_ops import Train_Collatefn
from ops.dist_ops import synchronize, all_reduce
from ops.log_ops import setup_logger
from ops.stat_ops import ScalarContainer, count_consective_num
from ops.acc_ops import topks_correct, topk_errors, topk_accuracies
from tqdm import tqdm
random.seed(0); torch.manual_seed(0); np.random.seed(0)
DIST_FLAG = torch.cuda.device_count() > 1

CFG_FILE = "cfgs/trainval_bc.yaml"

#fold_id = int(model_path.split('/')[-1][:2])

############### Set up Variables ###############
with open(CFG_FILE, "r") as f: cfg = yaml.safe_load(f)

TRAIN_DATA_ROOT = cfg["DATASETS"]["TRAIN_DATA_ROOT"]
VAL_DATA_ROOT = cfg["DATASETS"]["VAL_DATA_ROOT"]

MODEL_UID = cfg["MODEL"]["MODEL_UID"]
ARCH = cfg["MODEL"]["ARCH"]
DEPTH = cfg["MODEL"]["DEPTH"]
NUM_CLASSES = cfg["MODEL"]["NUM_CLASSES"]
NUM_WORKERS = int(cfg["DATALOADER"]["NUM_WORKERS"])
LOG_FILE = cfg["SOLVER"]["LOG_FILE"]
SNAPSHOT_MODEL_TPL = cfg["SOLVER"]["SNAPSHOT_MODEL_TPL"]
LEARNING_RATE = float(cfg["SOLVER"]["LEARNING_RATE"])
WEIGHT_DECAY = float(cfg["SOLVER"]["WEIGHT_DECAY"])
LR_DECAY = float(cfg["SOLVER"]["LR_DECAY"])
TRAIN_EPOCH = int(cfg["SOLVER"]["TRAIN_EPOCH"])
SNAPSHOT_FREQ = int(cfg["SOLVER"]["SNAPSHOT_FREQ"])
SNAPSHOT_HOME = cfg["SOLVER"]["SNAPSHOT_HOME"]
RESUME_EPOCH = int(cfg["SOLVER"]["RESUME_EPOCH"])
INIT_MODEL_PATH = cfg["SOLVER"]["INIT_MODEL_PATH"]
INIT_MODEL_STRICT = eval(cfg["SOLVER"]["INIT_MODEL_STRICT"])
RESUME_BOOL = bool(INIT_MODEL_PATH != "")

model = import_module(f"model.{MODEL_UID}")
ENModel = getattr(model, "ENModel")
criterion = torch.nn.CrossEntropyLoss(reduction="mean")

############### Set up Dataloaders ###############
Trainset = CTDataset(datalist=TRAIN_DATA_ROOT, target='train',)

Validset = CTDataset(datalist=VAL_DATA_ROOT, target='train')

model = ENModel(arch=ARCH, resnet_depth=DEPTH, 
                input_channel=1, num_classes=NUM_CLASSES)
model = model.cuda()

print(model)
############### Logging out some training info ###############
logger = setup_logger(logfile=LOG_FILE)
logger.info("Config {}...".format(CFG_FILE))
logger.info("{}".format(json.dumps(cfg, indent=1)))

TrainLoader = torch.utils.data.DataLoader(Trainset,
                                    batch_size=1, 
                                    num_workers=NUM_WORKERS,
                                    collate_fn=Train_Collatefn,
                                    shuffle=True)

ValidLoader = torch.utils.data.DataLoader(Validset,
                                    batch_size=1, 
                                    num_workers=NUM_WORKERS,
                                    collate_fn=Train_Collatefn,
                                    shuffle=False)
############### Set up Optimization ###############
optimizer = torch.optim.Adam([{"params": model.parameters(), "initial_lr": LEARNING_RATE}], 
                                                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scher = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY, last_epoch=RESUME_EPOCH)
criterion = torch.nn.CrossEntropyLoss(reduction="mean")

if INIT_MODEL_PATH != "":
    model.load_state_dict(torch.load(INIT_MODEL_PATH, \
                 map_location=f'cuda:{0}'), strict=INIT_MODEL_STRICT)

logger.info("Do train...")

############### Logging out some training info ###############
logger = setup_logger()
os.makedirs(SNAPSHOT_HOME, exist_ok=True)
logger = setup_logger(logfile=LOG_FILE)

dset_len, loader_len = len(Trainset), len(TrainLoader)
logger.info("Setting Dataloader | dset: {} / loader: {}, | N: {}".format(\
                            dset_len, loader_len, -1))

max_acc = 0
for e in range(RESUME_EPOCH, TRAIN_EPOCH):
    rT, all_tik = 0, time.time()        # rT -> run training time
    Epoch_CE, Epoch_Acc = [ScalarContainer() for _ in range(2)]
    for i, (all_F, all_L, all_info) in enumerate(tqdm(TrainLoader)):
        tik = time.time()
        optimizer.zero_grad()

        labels = all_L.cuda()
        inputs = all_F.cuda()

        # calculate loss and metrics
        train_loss, train_acc, _, _, _ = model.calculate_objective([inputs], labels)        
        train_loss.backward()
        optimizer.step()
        Epoch_CE.write(train_loss); Epoch_Acc.write(train_acc)
        rT += time.time()-tik

    dT = (time.time()-all_tik) - rT     # dT -> data loading time
    Ece, Epoch_Acc = Epoch_CE.read(), Epoch_Acc.read()

    logger.info("TRAIN | E-I [{}-{}] | CE: {:1.5f} | TrainAcc: {:1.3f} | dT/rT: {:.3f} / {:.3f}".format(e, loader_len, Ece, Epoch_Acc, dT, rT))

    if e % SNAPSHOT_FREQ == 0 or e >= TRAIN_EPOCH-1:
        model.eval()
        VAL_CE, VAL_Acc = [ScalarContainer() for _ in range(2)]
        logger.info("Do validation...")

        with torch.no_grad():
            for i, (all_F, all_L, all_info) in enumerate(tqdm(ValidLoader)):
                labels = all_L.cuda()
                inputs = all_F.cuda()
                val_loss, val_acc, _, _, _ = model.calculate_objective([inputs], labels)  
                VAL_CE.write(val_loss); VAL_Acc.write(val_acc)

        Ece, EAcc = VAL_CE.read(), VAL_Acc.read()
        logger.info("VALIDATION | E [{}] | CE: {:1.5f} | Acc: {:1.3f}".format(e, Ece, EAcc)) 

        if max_acc <= EAcc:
            max_acc = EAcc
            model_save_path = os.path.join(SNAPSHOT_HOME, SNAPSHOT_MODEL_TPL.format(e))
            logger.info (f"Dump weights {model_save_path} to disk...")
            torch.save(model.state_dict(), model_save_path)
        else:
            logger.info (f"early stopping...")
            break
        model.train()
        
    if LR_DECAY != 1:
        lr_scher.step()
        logger.info("Setting LR: {}".format(optimizer.param_groups[0]["lr"]))