"""
Load each patient's all/specfied CT images.
"""
import cv2
from torch.utils import data
from PIL import Image
import os
import torchvision.transforms.functional as TF
import numpy as np
import torch
import random
from scipy.ndimage import zoom

try:
    from ops.dataset_ops import Rand_Affine, Rand_Crop, Rand_Transforms
except:
    #print ("Import external...")
    import sys
    sys.path.insert(0, "..")
    from ops.dataset_ops import Rand_Affine, Rand_Crop, Rand_Transforms

readvdnames = lambda x: open(x).read().rstrip().split('\n')

class CTDataset(data.Dataset):
    def __init__(self, datalist="",
                       target="train",
                       crop_size=(256, 256),
                       logger=None):

        npy_files, labels = [], []
        with open(datalist, 'r') as f:
            for i in f.read().splitlines():
                npy_files.append(i.split(',')[0])
                labels.append(int(i.split(',')[1]))
        
        self.datalist = datalist
        self.labels = labels
        self.meta = npy_files
        self.crop_size = crop_size
        #print (self.meta)
        self.data_len = len(self.meta)
        self.target = target

    def __getitem__(self, index):
        npy_path, label = self.meta[index], self.labels[index]

        cta_images = np.load(npy_path)

        num_frames = len(cta_images)
        shape = cta_images.shape

        # Data augmentation
        if self.target == "train":
            cta_images = Rand_Transforms(cta_images, ANGLE_R=10, TRANS_R=0.1, SCALE_R=0.2, SHEAR_R=10, BRIGHT_R=0.5, CONTRAST_R=0.3)
            cta_images = cta_images / 255.
        # To Tensor and Resize
        #cta_images = np.asarray(cta_images, dtype=np.float32)
        #cv2.imwrite('test_5.jpg', cta_images[100])
        

        label = np.uint8([label])

        info = {"name": npy_path, "num_frames": num_frames, "shape": shape}

        th_img = torch.unsqueeze(torch.from_numpy(cta_images.copy()), 0).float()
        th_img = torch.unsqueeze(th_img, 0)
        th_label = torch.from_numpy(label.copy()).long()

        return th_img, th_label, info

    def __len__(self):
        return self.data_len

    def debug(self, index):
        import cv2
        th_img, th_label, info = self.__getitem__(0)
        # th_img: NxCxTxHxW
        print(th_img.shape)
        img, label = th_img.numpy()[0, 0, 100, :]*255, th_label.numpy()[0]
        #n, h, w = img.shape
        #print ("[DEBUG] Writing to {}".format(debug_f))
        print(label, info)
        cv2.imwrite('test_6.jpg', img)


if __name__ == "__main__":
    # Read valid sliding: 550 seconds
    ctd = CTDataset(datalist="/home/ubuntu/nas/projects/CTScreen/dataset/train-seg.text", target="val", crop_size=(256, 256))
    length = len(ctd)
    ctd[0]
    ctd.debug(0)
    #exit()
    # ctd.debug(0)
    # import time
    # s = time.time()
    # for i in range(length):
    #     print (i)
    #     th_img, th_label, info = ctd[i]
    # e = time.time()
    # print ("time: ", e-s)

    # images, labels, info = ctd[0]
    # for i in range(10):
    #    ctd.debug(i)
    # import pdb
    # pdb.set_trace()


