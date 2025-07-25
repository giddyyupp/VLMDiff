import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

from PIL import Image
import copy

import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from lavis.models import load_model_and_preprocess

# datasets and loaders
from data_loaders.mvtecad_dataloader import MVTecDataset
from data_loaders.visa_dataloader import VisaDataset
from data_loaders.cifar_dataloader import CifarAD
from data_loaders.coco_dataloader import DefaultAD
from data_loaders.realiad_dataloader import RealIAD

import utils.internvl_utils as intern_utils

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def load_vlm(model_type='blip'):
    """
    blip, llava and internvl are the options here
    """
    if model_type == 'blip':
        model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
        return {'model': model, "vis_processors": vis_processors}
    
    elif model_type == 'llava':
        from llava.model.builder import load_pretrained_model

        pretrained = "lmms-lab/llama3-llava-next-8b"
        model_name = "llava_llama3"
        # device = "cuda"
        device_map = "auto"
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None) # Add any other thing you want to pass in llava_model_args

        model.eval()
        model.tie_weights()
        return {'model': model, "tokenizer": tokenizer, "image_processor": image_processor, "max_length": max_length}

    elif model_type == 'internvl':
        from transformers import AutoModel, AutoTokenizer

        # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
        path = 'OpenGVLab/InternVL2-8B'
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        return {'model': model, "tokenizer": tokenizer}
    

def get_caption_blip(image, prompt, **kwargs):
    """
    image: PIL Image
    prompt: str
    """
    # prepare the image
    vis_processors = kwargs['vis_processors']
    model = kwargs['model']
    # print(model)

    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    response = model.generate({"image": image, "prompt": prompt})

    return response[0]


def get_caption_llava(image, prompt, **kwargs):
    """
    image: PIL Image
    prompt: str
    """
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates, SeparatorStyle
    
    model = kwargs['model']
    image_processor = kwargs['image_processor']
    tokenizer = kwargs['tokenizer']

    conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]   

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)
    # print(response)
    return response


def get_caption_internvl(image, prompt, **kwargs):
    model = kwargs['model']
    tokenizer = kwargs['tokenizer']
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    pixel_values = intern_utils.load_image(image, max_num=12).to(torch.bfloat16).cuda()

    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    # print(f'Assistant: {response}')
    return response


def run_capt(data_loader, prompt, captioner='blip', dataset_name='coco', save_dir='.', start_ind=0, end_ind=120000):

    captions, labels = [], []

    model_params = load_vlm(model_type=captioner)

    print(len(data_loader))

    for i, data_dict in enumerate(data_loader):
        if i < start_ind:
            continue
        if i >= end_ind:
            break

        print(i)
        # get the inputs; data is a list of [inputs, labels]
        if dataset_name in ['mvtec', 'visa', 'realiad']: # custom loaders
            image = data_dict['jpg']
            image_name = data_dict['filename']
            labels.append(image_name[0])
        elif dataset_name == 'coco': # coco
            image = data_dict[0]
            image_name = [str(data_loader.dataset.ids[i])]
            labels.append(image_name)
        elif dataset_name == 'cifar100': # cifar10-100
            image_name = [str(data_loader.dataset.targets[i])]
            labels.append(image_name)
            image = data_dict[0]

        image_pil = F.to_pil_image(image[0], mode='RGB')
        # plt.imshow(image_pil)
        # plt.savefig(f"im_{i}.png")

        if captioner == 'blip':
            response = get_caption_blip(image_pil, prompt, **model_params)

        elif captioner == 'llava':
            response = get_caption_llava(image_pil, prompt, **model_params)
        
        elif captioner == 'internvl':
            response = get_caption_internvl(image_pil, prompt, **model_params)

        # generate caption for this image and save it
        print(response, image_name[0])
        captions.append(response)
        # if (i+1) % 10000 == 0:
        #     with open(f"{save_dir}/test_captions_{dataset_name}_{caption_model}_{start_ind}_{i}.json", "w") as fp:
        #         json.dump(captions, fp)
        #     with open(f"{save_dir}/test_labels_{dataset_name}_{caption_model}_{start_ind}_{i}.json", "w") as fp:
        #         json.dump(labels, fp)
    
    return captions, labels


def get_cifar100(data_root, transform):
    
    # classes = ['plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    batch_size = 1

    trainset = torchvision.datasets.CIFAR100(root=data_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=data_root, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader


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

    parser = argparse.ArgumentParser(description="VLMDiff_captioner")
    parser.add_argument("--coco_part", default=0)
    parser.add_argument("--data_set", default='coco', help="choices are coco|cifar10|cifar100|mvtec|visa|realiad")
    parser.add_argument("--data_path", default='/mnt/isilon/projects/sigcom_cv/shared_datasets/coco')
    parser.add_argument("--vlm_model", default='internvl', help='which VLM to use. internvl|deepseekvl|blip')
    parser.add_argument("--prompt_type", default='pd', help='which prompt type to use,  pd|pa')
    parser.add_argument('--only_train_set', action='store_true', help='extract VLM desc only for training set, if not set runs for test too')

    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=10000)

    args = parser.parse_args()

    dataset_name = args.data_set
    data_path = args.data_path

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
    captions, labels = run_capt(train_dataloader, prompt, caption_model, dataset_name, start_ind=args.start_ind, end_ind=args.end_ind)
    with open(f"{save_dir}/train_captions_{dataset_name}_{caption_model}_{args.start_ind}_{args.end_ind}.json", "w") as fp:
       json.dump(captions, fp)
    with open(f"{save_dir}/train_labels_{dataset_name}_{caption_model}_{args.start_ind}_{args.end_ind}.json", "w") as fp:
       json.dump(labels, fp)

    # Test Set
    if not args.only_train_set:
        captions, labels = run_capt(test_dataloader, prompt, caption_model, dataset_name, save_dir)
        with open(f"{save_dir}/test_captions_{dataset_name}_{caption_model}_{args.start_ind}_{args.end_ind}.json", "w") as fp:
            json.dump(captions, fp)
        with open(f"{save_dir}/test_labels_{dataset_name}_{caption_model}_{args.start_ind}_{args.end_ind}.json", "w") as fp:
            json.dump(labels, fp)
