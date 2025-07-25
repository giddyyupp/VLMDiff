import os
import csv
import json
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from data_loaders.data_utils import data_transforms, gt_transforms, get_captions


class VisaDataset(Dataset):
	def __init__(self, root, train=True, use_captions=False, caption_folder='', data_tf=None, shuffle_data=True):
		self.data = []
		self.root = root
		self.train = train
		self.use_captions = use_captions
		self.image_size = (256,256)
		self.transform = data_transforms(self.image_size) if data_tf is None else data_tf
		self.target_transform = gt_transforms(self.image_size)
		self.caption_folder = caption_folder

		split = 'test'
		if train:
			split = 'train'

		with open('./training/VisA/visa.csv', 'rt') as f:
			render = csv.reader(f, delimiter=',')
			header = next(render)
			for row in render:
				if row[1] == split:
					data_dict = {'object':row[0],'split':row[1],'label':row[2],'image':row[3],'mask':row[4], 'caption': ['']}
					self.data.append(data_dict)
		self.label_to_idx = {'candle': '0', 'capsules': '1', 'cashew': '2', 'chewinggum': '3', 'fryum': '4', 'macaroni1': '5',
							 'macaroni2': '6', 'pcb1': '7', 'pcb2': '8', 'pcb3': '9', 'pcb4': '10',
							 'pipe_fryum': '11',}

		if use_captions:
			self.captions, self.labels = get_captions(self.root, train, self.caption_folder)
			
			for ind, caption in enumerate(self.captions):
				assert self.data[ind]['image'] == self.labels[ind]
				self.data[ind]['caption'] = [caption]

		self.cls_to_data_d = {}
		for lbl, v in self.label_to_idx.items():
			self.cls_to_data_d[lbl] = []

		for dd in self.data:
			self.cls_to_data_d[dd['object']].append(dd['image']) 

		if shuffle_data and self.train:
			random.shuffle(self.data)
		self.length = len(self.data)
	
	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		item = self.data[idx]
		source_filename = item['image']
		target_filename = item['image']

		if item.get("mask", None):
			mask = Image.open(os.path.join(self.root, item['mask'])).convert('L')
		else:
			if item['label'] == 'normal':  # good
				mask = np.zeros(self.image_size).astype(np.uint8)
			elif item['label'] == 'anomaly':  # defective
				mask = np.ones(self.image_size).astype(np.uint8)
			else:
				raise ValueError("Labels must be [None, 0, 1]!")

			mask = Image.fromarray(mask, "L") 

		caption = item['caption']

		source = Image.open(os.path.join(self.root, source_filename)).convert('RGB')
		target = Image.open(os.path.join(self.root, target_filename)).convert('RGB')

		source = self.transform(source)
		target = self.transform(target)
		mask = self.target_transform(mask)

		clsname = item["object"]
		image_idx = self.label_to_idx[clsname]

		return dict(jpg=target, txt=caption[0], hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))
