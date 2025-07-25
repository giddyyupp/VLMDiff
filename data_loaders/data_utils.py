import os
from torchvision import transforms
import json


mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def data_transforms(size):
    datatrans =  transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                            std=std_train),])
    return datatrans

def gt_transforms(size):
    gttrans =  transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),])
    return gttrans


def get_captions(root, train, caption_folder):
    train_capt_file = [entry for entry in os.listdir(os.path.join(root, '..', caption_folder)) if entry.startswith("train_captions")][0]
    train_label_file = [entry for entry in os.listdir(os.path.join(root, '..', caption_folder)) if entry.startswith("train_labels")][0]
    test_capt_file = [entry for entry in os.listdir(os.path.join(root, '..', caption_folder)) if entry.startswith("test_captions")][0]
    test_label_file = [entry for entry in os.listdir(os.path.join(root, '..', caption_folder)) if entry.startswith("test_labels")][0]
    if train:
        with open(os.path.join(root, '..', caption_folder, train_capt_file)) as ff:
            captions = json.load(ff)
        with open(os.path.join(root, '..', caption_folder, train_label_file)) as ff:
            labels = json.load(ff)
    else:
        with open(os.path.join(root, '..', caption_folder, test_capt_file)) as ff:
            captions = json.load(ff)
        with open(os.path.join(root, '..', caption_folder, test_label_file)) as ff:
            labels = json.load(ff)

    return captions, labels
