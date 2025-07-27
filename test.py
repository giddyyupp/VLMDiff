import os
import argparse
import numpy as np
import logging

from PIL import Image

from share import *

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import functional as F
import timm
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm

from sgn.model import create_model, load_state_dict
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.util import cal_anomaly_map, log_local, create_logger, setup_seed, get_dset_from_filename

from data_loaders.visa_dataloader import VisaDataset
from data_loaders.mvtecad_dataloader import MVTecDataset
from data_loaders.cifar_dataloader import CifarAD
from data_loaders.coco_dataloader import DefaultAD
from data_loaders.realiad_dataloader import RealIAD


def get_dino_attentions(model, img, patch_size=8, img_size=256, threshold=None):
    w_featmap = img_size // patch_size
    h_featmap = img_size // patch_size
    attentions = model.get_last_selfattention(img.cuda())

    nh = attentions.shape[1] # number of head
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]

    return [attentions.unsqueeze(0)]


parser = argparse.ArgumentParser(description="VLMDiff")
parser.add_argument("--resume_path", default='./ckpt_testing_mvtec/model_epoch=000.ckpt')
parser.add_argument("--coco_part", default=0)
parser.add_argument("--data_set", default='mvtec', help="choices are coco|cifar10|cifar100|mvtec|visa|realiad")
parser.add_argument("--data_path", default='/mnt/isilon/shicsonmez/ad/data/mvtec_anomaly_detection')
parser.add_argument("--exp_name", default='testing_mvtec', help='npz files will be saved inside this folder. make it unique please!')
parser.add_argument('--run_eval_only', action='store_true', help='runs just the eval part without generating images. concats npz_ header with exp_name param.')
parser.add_argument('--use_dino', action='store_true', help='use DINO as feature extractor or not, if not set Resnet-50.')
parser.add_argument("--dino_version", default='v1s8', help="choices are v1s8|v1s16|v1b8|v1b16|v2s14|v1r50")
parser.add_argument("--caption_folder", default='', help='which caption folder to use! for coco just use the vlm name')
parser.add_argument('--use_captions', action='store_true', help='use text descriptions as additional guidance or not')
parser.add_argument("--start_ind", type=int, default=0)
parser.add_argument("--end_ind", type=int, default=35000)
parser.add_argument('--save_visuals', action='store_true', help='save predictions visuals such as heatmaps, anomaly maps etc.')

args = parser.parse_args()

"""
python test.py --resume_path './val_ckpt_coco_part0_coco_part0_scratch_pretrained_ae/epoch=49-step=31749.ckpt' 
--coco_part 0 --data_set 'coco' --data_path '/project/home/p200249/shicsonmez/COCO' 
--exp_name 'coco_part1_scratch_pretrained_ae' --run_eval_only --use_dino --caption_folder '_internvl'
"""

# Configs
run_eval_only = args.run_eval_only
resume_path = args.resume_path
data_set = args.data_set
coco_part = args.coco_part
exp_name = args.exp_name
# save_dir = f"val_ckpt_{data_set}_{exp_name}"
meta_file = f'meta_20_{coco_part}.json'
evl_dir = f"npz_{args.exp_name}"
data_path = args.data_path  # '/mnt/isilon/shared_datasets/coco'
caption_folder = args.caption_folder  # for coco: '_blip' or '_internvl' or empty
dino_version = args.dino_version

batch_size = 1
logger_freq = 300
learning_rate = 1e-5
only_mid_control = True
use_captions = args.use_captions
use_dino = args.use_dino
run_count = 1  # for batch run put a bigger number like 20

logger = create_logger("global_logger", f"log_{exp_name}/")

print(args)

if not run_eval_only:
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('models/vlmdiff.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.only_mid_control = only_mid_control
    model.evl_dir = evl_dir

    # dataset prep
    if data_set == 'coco':
        caption_model = caption_folder  # '_deepseekvl'  # for coco: '_blip' or '_internvl' or empty
        test_dataset = DefaultAD(root=data_path, meta=meta_file, train=False, use_captions=use_captions, caption_model=caption_model, shuffle_data=False)
    elif data_set == 'mvtec':
        test_dataset = MVTecDataset(root=data_path, train=False, use_captions=use_captions, caption_folder=caption_folder, shuffle_data=False)
    elif data_set == 'visa':
        test_dataset = VisaDataset(root=data_path, train=False, use_captions=use_captions, caption_folder=caption_folder, shuffle_data=False)
    elif data_set == 'cifar10':
        test_dataset = CifarAD(root=data_path, train=False, use_captions=use_captions, cifar_type='cifar10')
    elif data_set == 'cifar100':
        test_dataset = CifarAD(root=data_path, train=False, use_captions=use_captions, cifar_type='cifar100')
    elif data_set == 'realiad':
        test_dataset = RealIAD(root=data_path, train=False, use_captions=use_captions, caption_folder=caption_folder, shuffle_data=False)
    else:
        raise Exception("Not supported dataset type!!")

    # Misc
    test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=batch_size, shuffle=False)

    if use_dino:
        if dino_version == 'v2s14':
            # DINOv2
            pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            patch_size = 14
            img_size = 224
        # DINOv1
        elif dino_version == 'v1s8':
            pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')  # best performant!!
            patch_size = 8
            img_size = 256
        elif dino_version == 'v1s16':
            pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')  # best performant!!
            patch_size = 16
            img_size = 256
        elif dino_version == 'v1b8':
            pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
            patch_size = 8
            img_size = 256
        elif dino_version == 'v1b16':
            pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')  # best performant!!
            patch_size = 16
            img_size = 256
        elif dino_version == 'v1r50':
            # DINO R50!
            from torchvision.models.feature_extraction import create_feature_extractor
            return_nodes = {
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4': 'layer4',
            }
            pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            pretrained_model = create_feature_extractor(pretrained_model, return_nodes=return_nodes)
            patch_size = 8
            img_size = 256

        pretrained_model = pretrained_model.cuda()
        pretrained_model.eval()
    else:  # imagenet R50 model
        pretrained_model = timm.create_model("resnet50", pretrained=True, features_only=True)
        pretrained_model = pretrained_model.cuda()
        pretrained_model.eval()
    
    for run_id in range(run_count):
        root = os.path.join(f'log_image_{exp_name}_test_id_{run_id}/')
        log_dir = os.path.join(f'log_image_{exp_name}_id_{run_id}_visual/')
        os.makedirs(root, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        model.eval()
        os.makedirs(evl_dir, exist_ok=True)
        model = model.cuda()
        print(len(test_dataloader))

        start_ind = args.start_ind  # 30000
        end_ind = args.end_ind  # 60000

        # fileinfos = []
        # preds = []
        # masks = []
        with torch.no_grad():
            for i, input in tqdm(enumerate(test_dataloader)):
                if i < start_ind:
                    continue
                if i >= end_ind:
                    break
                input_img = input['jpg']
                # feature extraction for input image
                if use_dino:
                    input_img_dino = input_img.clone()
                    if 'r50' in dino_version:
                        # DINO R50 
                        input_features = pretrained_model.forward(input_img_dino.cuda())
                        input_features = [v for k, v in input_features.items()]
                        input_features = input_features[1:4]
                    elif 'v1' in dino_version:
                        # patch features
                        input_features = [pretrained_model.get_intermediate_layers(input_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, 1:, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
                    elif 'v2' in dino_version:
                        # Dinov2
                        input_features = [pretrained_model.get_intermediate_layers(input_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, :, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
                        # attention maps, not working very well
                        # input_features = get_dino_attentions(pretrained_model, input_img_dino)
                else:
                    # RESNET-50
                    input_features = pretrained_model(input_img.cuda())

                # forward image to model
                output= model.log_images_test(input)
                images = output
                # log generated image
                file_name = input["filename"][0]
                mid_path, name = get_dset_from_filename(file_name)
                
                # Note: loggin images: input and reconstruction
                if args.save_visuals:
                    log_local(images, file_name, log_dir=log_dir)
                output_img = images['samples']

                # feature extraction for reconstructed image
                if use_dino:
                    output_img_dino = output_img.clone()
                    if 'r50' in dino_version:
                        # DINO R50 
                        output_features = pretrained_model.forward(output_img_dino.cuda())
                        output_features = [v for k, v in output_features.items()]
                        output_features = output_features[1:4]
                    elif 'v1' in dino_version:
                        # patch features
                        output_features = [pretrained_model.get_intermediate_layers(output_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, 1:, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
                    elif 'v2' in dino_version:
                        # Dinov2 patch
                        output_features = [pretrained_model.get_intermediate_layers(output_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, :, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
                        # attention maps, not working well
                        # output_features = get_dino_attentions(pretrained_model, output_img_dino)

                else:
                    output_features = pretrained_model(output_img.cuda())

                if not use_dino:
                    input_features = input_features[1:4]
                    output_features = output_features[1:4]

                # Calculate the anomaly score
                anomaly_map, _ = cal_anomaly_map(input_features, output_features, input_img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=5)
                anomaly_map = torch.from_numpy(anomaly_map)
                anomaly_map_prediction = anomaly_map.unsqueeze(dim=0).unsqueeze(dim=1)

                filename_feature = "{}-features.jpg".format(name)
                path_feature = os.path.join(root, mid_path, filename_feature)
                os.makedirs(os.path.dirname(path_feature), exist_ok=True)
                pred_feature = anomaly_map_prediction.squeeze().detach().cpu().numpy()
                pred_feature = (pred_feature * 255).astype("uint8")
                pred_feature = Image.fromarray(pred_feature, mode='L')
                if args.save_visuals:
                    pred_feature.save(path_feature)

                #Heatmap
                anomaly_map_new = np.round(255 * (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min()))
                anomaly_map_new = anomaly_map_new.cpu().numpy().astype(np.uint8)
                heatmap = cv2.applyColorMap(anomaly_map_new, colormap=cv2.COLORMAP_JET)
                pixel_mean = [0.485, 0.456, 0.406]
                pixel_std = [0.229, 0.224, 0.225]
                pixel_mean = torch.tensor(pixel_mean).unsqueeze(1).unsqueeze(1)  # 3 x 1 x 1
                pixel_std = torch.tensor(pixel_std).unsqueeze(1).unsqueeze(1)
                image = (input_img.squeeze() * pixel_std + pixel_mean) * 255
                image = image.permute(1, 2, 0).to('cpu').numpy().astype('uint8')
                image_copy = image.copy()
                out_heat_map = cv2.addWeighted(heatmap, 0.5, image_copy, 0.5, 0, image_copy)
                heatmap_name = "{}-heatmap.png".format(name)
                if args.save_visuals:
                    cv2.imwrite(os.path.join(root, mid_path, heatmap_name), out_heat_map)

                input['pred'] = anomaly_map_prediction
                input["output"] = output_img.cpu()
                input["input"] = input_img.cpu()

                output2 = input
                dump(evl_dir, output2)
                # instead of dumping and loading lets just append here
                # fileinfos.append(
                #     {
                #         "filename": str(input["filename"]),
                #         # "height": npz["height"],
                #         # "width": npz["width"],
                #         "clsname": str(input["clsname"]),
                #     }
                # )
                # preds.append(input["pred"])
                # masks.append(input["mask"])
            
            # preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
            # masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W     
            
        if start_ind == 0:
            print(evl_dir)
            evl_metrics = {'auc': [ {'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'appx'}, {'name': 'apsp'}, {'name': 'f1px'}, {'name': 'f1sp'}]}
            print("Gathering final results ...")
            fileinfos, preds, masks = merge_together(evl_dir)
            ret_metrics = performances(fileinfos, preds, masks, evl_metrics)
            log_metrics(ret_metrics, evl_metrics)
        else:
            print("Testing for the part is done...")

else:
    print(evl_dir)
    evl_metrics = {'auc': [ {'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'appx'}, {'name': 'apsp'}, {'name': 'f1px'}, {'name': 'f1sp'}]}
    print("Gathering final results ...")
    fileinfos, preds, masks = merge_together(evl_dir)
    ret_metrics = performances(fileinfos, preds, masks, evl_metrics)
    log_metrics(ret_metrics, evl_metrics)
