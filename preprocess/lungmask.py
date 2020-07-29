import numpy as np
import torch
import utils
import SimpleITK as sitk
from resunet import UNet
import warnings
import sys
from tqdm import tqdm
import cv2

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore",category=UserWarning)

# stores urls and number of classes of the models
model_urls = {('unet','R231'): ('https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth',3),
('unet','LTRCLobes'): ('https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth',6),
('unet','R231CovidWeb'): ('https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth',3)}


def apply(image, model=None, force_cpu=False, batch_size=20, volume_postprocessing=True, show_process=True):

    if model is None:
        model = get_model('unet', 'R231')

    voxvol = np.prod(image.GetSpacing())
    inimg_raw = sitk.GetArrayFromImage(image)
    del image

    if force_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.info("No GPU support available, will use CPU. Note, that this is significantly slower!")
            batch_size = 1
            device = torch.device('cpu')
    model.to(device)

    tvolslices, xnew_box = utils.preprocess(inimg_raw, resolution=[256, 256])
    
    tvolslices[tvolslices > 600] = 600
    tvolslices = np.divide((tvolslices+1024),1624) #TODO:Normalize
    
    torch_ds_val = utils.LungLabelsDS_inf(tvolslices)
    #TODO: save_raw_image
    # print(tvolslices.max())
    # re = np.squeeze(torch_ds_val)
    # output_path = "/home/ubuntu/nas/projects/CTScreen/dataset/raw.jpg"
    # cv2.imwrite(output_path, re*255)

    dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    timage_res = np.empty((np.append(0,tvolslices[0].shape)), dtype=np.uint8)

    with torch.no_grad():
        for X in dataloader_val:
            X = X.float().to(device)
            prediction = model(X)
            pls = torch.max(prediction,1)[1].detach().cpu().numpy().astype(np.uint8)
            timage_res = np.vstack((timage_res, pls))

    # postprocessing includes removal of small connected components, hole filling and mapping of small components to
    # neighbors
    
    if volume_postprocessing:
        outmask = utils.postrocessing(timage_res, 250/voxvol)
    else:
        outmask = timage_res
    #print(outmask.shape)
    #outmask = np.asarray([utils.reshape_mask(outmask[i], xnew_box[i], tvolslices.shape[1:]) for i in range(outmask.shape[0])], dtype=np.uint8)
    #TODO save processed img and mask as 256*256
    outmask = np.asarray(outmask, dtype=np.uint8)
    torch_ds_val = np.asarray(torch_ds_val)
    return outmask, torch_ds_val


def get_model(modeltype, modelname):
    model_url,n_classes = model_urls[(modeltype, modelname)]
    state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device('cpu'))
    if modeltype == 'unet':
        model = UNet(n_classes=n_classes, padding=True,  depth=5, up_mode='upsample', batch_norm=True, residual=False)
    elif modeltype == 'resunet':
        model = UNet(n_classes=n_classes, padding=True,  depth=5, up_mode='upsample', batch_norm=True, residual=True)
    else:
        logging.exception(f"Model {modelname} not known")
    model.load_state_dict(state_dict)
    model.eval()
    return model

