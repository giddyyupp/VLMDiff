import os
import json
from tqdm import tqdm
import numpy as np
import argparse

from PIL import Image
import copy
import torchvision.transforms.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from lmdeploy import pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# datasets and loaders
from torch.utils.data import DataLoader
from data_loaders.mvtecad_dataloader import MVTecDataset
from data_loaders.visa_dataloader import VisaDataset
from data_loaders.cifar_dataloader import CifarAD
from data_loaders.coco_dataloader import DefaultAD
from data_loaders.realiad_dataloader import RealIAD

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def run_capt(data_loader, prompt, root, dataset_name='coco'):

    captions, labels = [], []

    pipe = pipeline('deepseek-ai/deepseek-vl-1.3b-chat')

    print(len(data_loader))

    for i, data_dict in tqdm(enumerate(data_loader, 0)):
        print(i)
        # get the inputs; data is a list of [inputs, labels]
        if dataset_name in ['mvtec', 'visa', 'realiad']: # custom loaders
            image = data_dict['jpg']
            image_name = data_dict['filename']
            labels.append(image_name[0])
        elif dataset_name == 'coco': # coco
            image = data_dict[0]
            image_name = [str(data_loader.dataset.ids[i])]
            labels.append(image_name[0])
            image_name = [str(image_name[0]).zfill(12) + '.jpg']
            root = data_loader.dataset.root

        # image_pil = F.to_pil_image(image[0], mode='RGB')
        # plt.imshow(image_pil)
        # plt.savefig(f"im_{i}.png")

        image = load_image(os.path.join(root, image_name[0]))
        response = pipe((f'{prompt} {IMAGE_TOKEN}', image))

        # generate caption for this image and save it
        print(response.text, image_name[0])
        captions.append(response.text)
    
    return captions, labels


def get_mvtec(data_root, transform):
    
    train_dataset, test_dataset = MVTecDataset(data_root, train=True, data_tf=transform, shuffle_data=False), MVTecDataset(data_root, train=False, data_tf=transform, shuffle_data=False)

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=1, shuffle=False)   

    return train_dataloader, test_dataloader


def get_visa(data_root, transform):
    train_dataset, test_dataset = VisaDataset(data_root, train=True, data_tf=transform, shuffle_data=False), VisaDataset(data_root, train=False, data_tf=transform, shuffle_data=False)

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=1, shuffle=False)   

    return train_dataloader, test_dataloader


def get_realiad(data_root, transform):
    train_dataset, test_dataset = RealIAD(data_root, train=True, data_tf=transform, shuffle_data=False), RealIAD(data_root, train=False, data_tf=transform, shuffle_data=False)

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=1, shuffle=False)   

    return train_dataloader, test_dataloader


def get_coco(data_root, transform):

    train_dataset = torchvision.datasets.coco.CocoDetection(f'{data_root}train2017', annFile=f'{data_root}annotations/instances_train2017.json', transform=transform)
    test_dataset = torchvision.datasets.coco.CocoDetection(f'{data_root}val2017', annFile=f'{data_root}annotations/instances_val2017.json', transform=transform)

    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=False)   

    return train_dataloader, test_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VLMDiff_captioner_lmdeploy")
    parser.add_argument("--coco_part", default=0)
    parser.add_argument("--data_set", default='coco', help="choices are coco|mvtec|visa|realiad")
    parser.add_argument("--data_path", default='./coco')
    parser.add_argument("--vlm_model", default='deepseekvl', help='which VLM to use. internvl|deepseekvl|blip')
    parser.add_argument("--prompt_type", default='pd', help='which prompt type to use, pd|pa')
    parser.add_argument('--only_train_set', action='store_true', help='extract VLM desc only for training set, if not set, runs for test too.')

    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=1000000)

    args = parser.parse_args()

    dataset_name = args.data_set
    data_path = args.data_path
    root = data_path

    capt_transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == 'coco':
        train_dataloader, test_dataloader = get_coco(data_path, capt_transform)
    elif dataset_name == 'mvtec':
        train_dataloader, test_dataloader = get_mvtec(data_path, capt_transform)
    elif dataset_name == 'visa':
        train_dataloader, test_dataloader = get_visa(data_path, capt_transform)
    elif dataset_name == 'realiad':
        train_dataloader, test_dataloader = get_realiad(data_path, capt_transform)
    else:
        raise Exception("Not supported dataset type!!")

    prompt_coco = 'Describe the visual features of image in detail, with around 60 words and maybe 4 or 5 sentences.'  # internvl prompt
    prompt_pa = 'Be a pixel level visual defect inspector, and describe whether there is a defect or not on the main object itself, with around 60 words and maybe 4 or 5 sentences.'  # internvl prompt
    prompt_pd = 'Describe the visual features of image in detail with a focus on the main object, with around 60 words and maybe 4 or 5 sentences.'  # internvl prompt
    
    prompt = prompt_pd  # visa_internvl

    if dataset_name == 'coco':
        prompt = prompt_coco
    elif args.prompt_type == 'pd':
        prompt = prompt_pd
    else:
        prompt = prompt_pa

    caption_model = args.vlm_model  # 'internvl'
    save_dir = f"{dataset_name}_captions_{caption_model}_{args.prompt_type}"
    os.makedirs(save_dir, exist_ok=True)

    # Train Set
    captions, labels = run_capt(train_dataloader, prompt, root, dataset_name)
    with open(f"{save_dir}/train_captions_{dataset_name}_{caption_model}.json", "w") as fp:
        json.dump(captions, fp)
    with open(f"{save_dir}/train_labels_{dataset_name}_{caption_model}.json", "w") as fp:
        json.dump(labels, fp)

    # Test Set
    captions, labels = run_capt(test_dataloader, prompt, root, dataset_name)
    with open(f"{save_dir}/test_captions_{dataset_name}_{caption_model}.json", "w") as fp:
        json.dump(captions, fp)
    with open(f"{save_dir}/test_labels_{dataset_name}_{caption_model}.json", "w") as fp:
        json.dump(labels, fp)
