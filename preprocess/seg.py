import sys
import argparse
import logging
import lungmask
import utils
import os
import SimpleITK as sitk
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
# import scipy.misc
#from PIL import Image
def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')

def makedirs(path):
    if not os.path.exists(path):
	    os.makedirs(path)

def main(model, input):

    parser = argparse.ArgumentParser()
    #parser.add_argument('input', metavar='input', type=path, help='Path to the input image, can be a folder for dicoms')
    #parser.add_argument('output', metavar='output', type=str, help='Filepath for output lungmask')
    parser.add_argument('--modeltype', help='Default: unet', type=str, choices=['unet', 'resunet'], default='unet')
    parser.add_argument('--modelname', help="spcifies the trained model, Default: R231", type=str, choices=['R231','LTRCLobes','R231CovidWeb'], default='R231')
    parser.add_argument('--cpu', help="Force using the CPU even when a GPU is available, will override batchsize to 1", action='store_true')
    parser.add_argument('--nopostprocess', help="Deactivates postprocessing (removal of unconnected components and hole filling", action='store_true')
    parser.add_argument('--batchsize', type=int, help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.", default=100)

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    input_image = utils.get_input_image(input)
    #logging.info(f'Infer lungmask')
    mask, img = lungmask.apply(input_image, model, force_cpu=args.cpu, batch_size=batchsize,volume_postprocessing=not(args.nopostprocess))

    #mask = np.squeeze(mask)
    img = np.squeeze(img, axis = 0)
    #print(result.shape, img.shape)
    #print(result.max(), img.max())
    seg_img = np.where(mask<1, 0, img)
    #cv2.imwrite(img_save_path, seg_img*255)
    #write to path
    #cv2.imwrite(result_save_path, result*255)
    #cv2.imwrite(img_save_path, img*255)
    #result_out= sitk.GetImageFromArray(result)
    #result_out.CopyInformation(input_image)
    #print(result_out)
    #logging.info(f'Save result to: {output}')
    #sys.exit(sitk.WriteImage(result_out, output))
    return seg_img, img, mask


if __name__ == "__main__":

    logging.info(f'Load model')
    model = lungmask.get_model('unet', 'R231')

    train_file = "/home/ubuntu/nas/projects/AD3DMIL/dataset/train.txt"
    val_file = "/home/ubuntu/nas/projects/AD3DMIL/dataset/val.txt"
    test_file = "/home/ubuntu/nas/projects/AD3DMIL/dataset/test.txt"

    save_path = '/home/ubuntu/nas/projects/CTScreen/dataset/Seg/'

    def seg_dicom(train_file):
        logging.info(f'Processing List:{train_file}')
        with open(train_file, 'r') as f:
            file_dirs, labels = [], []
            for i in f.read().splitlines():
                file_dirs.append(i.split(',')[0])
                labels.append(int(i.split(',')[1]))
        save_txt_file = train_file.replace('.txt', '-seg.text')
        segimgfiles = []
        for index, file_dir in enumerate(file_dirs):
            dcm_files = glob(file_dir)
            #make sure npy save path
            if labels[index] == 1:
                npy_save_path = file_dir.replace('/home/ubuntu/nas/datasets/COVID-CT/COVID-19/', save_path)
                target_path = npy_save_path.split('/IMG')[0]
                segimg_save_path = npy_save_path.replace('*.dcm', 'segimg.npy')
                img_save_path = npy_save_path.replace('*.dcm', 'img.npy')
                mask_save_path = npy_save_path.replace('*.dcm', 'mask.npy')
            elif file_dir.split('/')[-4] == "SDTCM_Hosptial":
                npy_save_path = file_dir.replace('/home/ubuntu/nas/datasets/SDTCM_Hosptial/', save_path)
                #print(npy_save_path)
                target_path = npy_save_path.split('/*')[0]
                segimg_save_path = npy_save_path.replace('*.dcm', 'segimg.npy')
                img_save_path = npy_save_path.replace('*.dcm', 'img.npy')
                mask_save_path = npy_save_path.replace('*.dcm', 'mask.npy')
            else:
                npy_save_path = file_dir.replace('/home/ubuntu/nas/datasets/COVID-CT/Non-COVID-19/', save_path)
                target_path = npy_save_path.split('/*')[0]
                segimg_save_path = npy_save_path.replace('*.dcm', 'segimg.npy')
                img_save_path = npy_save_path.replace('*.dcm', 'img.npy')
                mask_save_path = npy_save_path.replace('*.dcm', 'mask.npy')

            segimgfiles.append(segimg_save_path)
            # if os.path.exists(target_path):
            #     logging.info(f'{target_path} exists!')
            #     continue 
            if os.path.isfile(segimg_save_path):
                logging.info(f'{segimg_save_path} exists!')
                continue 
            else:
                makedirs(target_path)
            segimg_3d = np.empty((np.append(0, (256, 256))), dtype=np.float)
            img_3d = np.empty((np.append(0, (256, 256))), dtype=np.float)
            mask_3d = np.empty((np.append(0, (256, 256))), dtype=np.uint8)
            logging.info(f'Processing CTs: {file_dir}')
            logging.info(f'Infer lungmask')
            for i, f in enumerate(tqdm(dcm_files)):
                input_file = f
                segimg, img, mask = main(model, input_file)
                segimg_3d = np.vstack((segimg_3d, segimg))
                img_3d = np.vstack((img_3d, img))
                mask_3d = np.vstack((mask_3d, mask))
            logging.info(f'Save results: {segimg_save_path}')
            np.save(segimg_save_path, segimg_3d)
            np.save(img_save_path, img_3d)
            np.save(mask_save_path, mask_3d)
        logging.info(f'Save list to: {save_txt_file}')
        with open(save_txt_file, 'w') as f:
            for i, d in enumerate(segimgfiles):
                f.write('{},{}\n'.format(d, labels[i]))
    seg_dicom(train_file)
    seg_dicom(val_file)
    seg_dicom(test_file)
