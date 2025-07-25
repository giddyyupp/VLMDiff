import os
import json
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from data_loaders.data_utils import data_transforms, gt_transforms, get_captions


class MVTecDataset(Dataset):
	def __init__(self, root, train=True, use_captions=False, caption_folder='', data_tf=None, shuffle_data=True):

		self.data = []
		self.root = root
		self.train = train
		self.use_captions = use_captions
		self.image_size = (256, 256)
		self.transform = data_transforms(self.image_size) if data_tf is None else data_tf
		self.target_transform = gt_transforms(self.image_size)
		self.caption_folder = caption_folder
	
		if train:
			with open('./training/MVTec-AD/train.json', 'rt') as f:
				for ind, line in enumerate(f):
					self.data.append(json.loads(line))
					self.data[ind].update({"caption": [""]})
		else:
			with open('./training/MVTec-AD/test.json', 'rt') as f:
				for ind, line in enumerate(f):
					self.data.append(json.loads(line))
					self.data[ind].update({"caption": [""]})

		self.label_to_idx = {'bottle': '0', 'cable': '1', 'capsule': '2', 'carpet': '3', 'grid': '4', 'hazelnut': '5',
							 'leather': '6', 'metal_nut': '7', 'pill': '8', 'screw': '9', 'tile': '10',
							 'toothbrush': '11', 'transistor': '12', 'wood': '13', 'zipper': '14'}

		if self.use_captions:
			self.captions, self.labels = get_captions(self.root, train, self.caption_folder)

			for ind, caption in enumerate(self.captions):
				assert self.data[ind]['filename'] == self.labels[ind]
				self.data[ind]['caption'] = [caption]
		
		self.cls_to_data_d = {}
		for lbl, v in self.label_to_idx.items():
			self.cls_to_data_d[lbl] = []

		for dd in self.data:
			self.cls_to_data_d[dd['clsname']].append(dd['filename']) 

		if shuffle_data and self.train:
			random.shuffle(self.data)
		self.length = len(self.data)

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		item = self.data[idx]
		source_filename = item['filename']
		target_filename = item['filename']
		label = item["label"]
		if item.get("maskname", None):
			mask = Image.open(os.path.join(self.root, item['maskname'])).convert('L')
		else:
			if label == 0:  # good
				mask = np.zeros(self.image_size).astype(np.uint8)
			elif label == 1:  # defective
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

		clsname = item["clsname"]
		image_idx = self.label_to_idx[clsname]

		return dict(jpg=target, txt=caption[0], hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))
