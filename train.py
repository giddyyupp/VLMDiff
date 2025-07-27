import os
import argparse

from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from sgn.logger import ImageLogger
from sgn.model import create_model, load_state_dict

from utils.util import setup_seed
from data_loaders.mvtecad_dataloader import MVTecDataset
from data_loaders.visa_dataloader import VisaDataset
from data_loaders.cifar_dataloader import CifarAD
from data_loaders.coco_dataloader import DefaultAD
from data_loaders.realiad_dataloader import RealIAD


parser = argparse.ArgumentParser(description="VLMDiffTrain")

parser.add_argument("--resume_path", default='./models/vlmdiff_sd15_model.ckpt', help='resume from checkpoint')
parser.add_argument("--coco_part", default=-1, help='there are 4 parts in coco dataset, 0 to 3. which one to train.')
parser.add_argument("--data_set", default='mvtec', help="choices are coco|realiad|mvtec|visa")
parser.add_argument("--data_path", default='/path/to/mvtec_dataset')
parser.add_argument("--exp_name", default='mvtec_training_vlmdiff', help='checkpoints will be saved inside this folder. make it unique please!')
parser.add_argument('--use_captions', action='store_true', help='use text descriptions as additional guidance or not')
parser.add_argument("--caption_folder", default='', help='which caption folder to load the extracted captions!')
parser.add_argument("--caption_model", default='_internvl', help='for coco only, choices are ["_blip", "_internvl", "_deepseekvl", ""]')

args = parser.parse_args()

print(args)

"""
python train.py --resume_path './models/coco_part0_pretrained_ae.ckpt' 
--coco_part 0 --data_set 'coco' --data_path '/project/home/p200249/shicsonmez/COCO' 
--exp_name 'coco_part0_scratch_pretrained_ae' --use_captions --caption_folder 'abc'
"""

eval_per_epochs = 25

data_set = args.data_set
coco_part = args.coco_part
exp_name = args.exp_name
save_dir = f"ckpt_{exp_name}"
meta_file = f'meta_20_{coco_part}.json'
evl_dir = f"npz_{args.exp_name}"
data_path = args.data_path
use_captions = args.use_captions
caption_folder = args.caption_folder

setup_seed(1)
batch_size = 12
logger_freq = 3000000000000
learning_rate = 1e-5
only_mid_control = True

# dataset prep
if data_set == 'coco':
    eval_per_epochs = 20
    save_dir = f"val_ckpt_{data_set}_part{coco_part}_{exp_name}"
    caption_model = args.caption_model  # for coco: '_blip' or '_internvl' or '_deepseekvl' or ''
    train_dataset = DefaultAD(root=data_path, meta=meta_file, train=True, use_captions=use_captions, caption_model=caption_model)
    test_dataset = DefaultAD(root=data_path, meta=meta_file, train=False, use_captions=use_captions, caption_model=caption_model)
elif data_set == 'mvtec':
    train_dataset = MVTecDataset(root=data_path, train=True, use_captions=use_captions, caption_folder=caption_folder)
    test_dataset = MVTecDataset(root=data_path, train=False, use_captions=use_captions, caption_folder=caption_folder)
elif data_set == 'visa':
    train_dataset = VisaDataset(root=data_path, train=True, use_captions=use_captions, caption_folder=caption_folder)
    test_dataset = VisaDataset(root=data_path, train=False, use_captions=use_captions, caption_folder=caption_folder)
elif data_set == 'realiad':
    eval_per_epochs = 20
    train_dataset = RealIAD(root=data_path, train=True, use_captions=use_captions, caption_folder=caption_folder)
    test_dataset = RealIAD(root=data_path, train=False, use_captions=use_captions, caption_folder=caption_folder)
else:
    raise Exception("Not supported dataset type!!")

# dataloaders
train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=1, shuffle=True)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/vlmdiff.yaml').cpu()
model.load_state_dict(load_state_dict(args.resume_path, location='cpu'),strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control
model.evl_dir = evl_dir

# trainer
# ckpt_callback_val_loss = ModelCheckpoint(monitor='val_acc', dirpath=save_dir, mode='max')
ckpt_callback_val_loss = ModelCheckpoint(
        filename="model_{epoch:03d}",
        every_n_epochs=eval_per_epochs,
        dirpath=save_dir,
        save_top_k=-1,  # <--- this is important!
        save_on_train_epoch_end=True,
    )
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, ckpt_callback_val_loss], accumulate_grad_batches=4, check_val_every_n_epoch=eval_per_epochs, limit_val_batches=0)

# Train!
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
