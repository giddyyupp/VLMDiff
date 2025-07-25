import os
import cv2
import argparse
import numpy as np
import torch
import torchvision
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import instantiate_from_config

from data_loaders.mvtecad_dataloader import MVTecDataset
from data_loaders.visa_dataloader import VisaDataset
from data_loaders.cifar_dataloader import CifarAD
from data_loaders.coco_dataloader import DefaultAD
from data_loaders.realiad_dataloader import RealIAD


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join('/mnt/isilon/shicsonmez/ad/repos/VLMDiff', "image_log_coco2", split)
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        pixel_mean = torch.tensor(pixel_mean).unsqueeze(1).unsqueeze(1)  # 3 x 1 x 1
        pixel_std = torch.tensor(pixel_std).unsqueeze(1).unsqueeze(1)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid.squeeze() * pixel_std + pixel_mean) * 255
            grid = grid.permute(1, 2, 0).to('cpu').numpy()
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, grid)
            # Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if True:
            # (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
            #     hasattr(pl_module, "log_images") and
            #     callable(pl_module.log_images) and
            #     self.max_images > 0):
            # logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    # if self.clamp:
                    #     images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            # if is_train:
            #     pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % trainer.check_val_every_n_epoch ==0:
            if not self.disabled:
                self.log_img(pl_module, batch, batch_idx, split="train")

def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict
def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model



parser = argparse.ArgumentParser(description="VLMDiffTrainAE")

parser.add_argument("--resume_path", default='./models/autoencoders.ckpt', help='resume from checkpoint')
parser.add_argument("--coco_part", default=0, help='there are 4 parts in coco dataset, 0 to 3. which one to train.')
parser.add_argument("--data_set", default='coco', help="choices are coco|cifar10|cifar100|mvtec|visa")
parser.add_argument("--data_path", default='/path/to/dataset')

args = parser.parse_args()

print(args)

data_set = args.data_set
coco_part = args.coco_part
meta_file = f'meta_20_{coco_part}.json'
data_path = args.data_path

# Configs
resume_path = args.resume_path   # autoencoders

batch_size = 1
logger_freq = 3000000000000
learning_rate = 4.5e-5
sd_locked = False
only_mid_control = True

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/autoencoder_kl_32x32x4.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
model.learning_rate = batch_size * learning_rate

# dataset prep
if data_set == 'coco':
    # save_dir = f"val_ckpt_{data_set}_part{coco_part}_{exp_name}"
    # caption_model = args.caption_model  # for coco: '_blip' or '_internvl' or '_deepseekvl' or ''
    train_dataset = DefaultAD(root=data_path, meta=meta_file, train=True, use_captions=False, caption_model='')
elif data_set == 'mvtec':
    train_dataset = MVTecDataset(root=data_path, train=True, use_captions=False, caption_folder='')
elif data_set == 'visa':
    train_dataset = VisaDataset(root=data_path, train=True, use_captions=False, caption_folder='')
elif data_set == 'realiad':
    train_dataset = RealIAD(root=data_path, train=True, use_captions=False, caption_folder='')
else:
    raise Exception("Not supported dataset type!!")

# create the dataloader
train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq, log_first_step=True)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger],  accumulate_grad_batches=8)

# Train!
trainer.fit(model, train_dataloaders=train_dataloader)

# vis autoencoder, change the resume_path to a trained model to visualise a dataset
# model.logger = logger
# for ind, dd in enumerate(train_dataloader):
#     logger.log_img(model, dd, ind, split="train")

