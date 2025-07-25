import os
import json
from PIL import Image
import random
import numpy as np

from torch.utils.data import Dataset

from data_loaders.data_utils import data_transforms, gt_transforms, get_captions


class RealIAD(Dataset):
	def __init__(self, root, train=True, use_captions=False, caption_folder='', data_tf=None, shuffle_data=True):
		self.root = root
		self.train = train
		self.use_captions = use_captions
		self.caption_folder = caption_folder

		self.image_size = (256,256)
		self.transform = data_transforms(self.image_size) if data_tf is None else data_tf
		self.target_transform = gt_transforms(self.image_size)

		cls_names = os.listdir(self.root)
		real_cls_names = []
		for cls_name in cls_names:
			if cls_name.split('.')[0] not in real_cls_names:
				real_cls_names.append(cls_name.split('.')[0])
		real_cls_names.sort()
		self.cls_names = real_cls_names

		meta_info = dict()
		for cls_name in self.cls_names:
			data_cls_all = []
			cls_info = json.load(open(f'{self.root}/../real_iad_jsons/realiad_jsons/{cls_name}.json', 'r'))
			data_cls = cls_info['train' if self.train else 'test']
			for data in data_cls:
				if data['anomaly_class'] == 'OK':
					info_img = dict(
						img_path=f"{cls_name}/{data['image_path']}",
						mask_path='',
						cls_name=cls_name,
						specie_name='',
						anomaly=0,
						caption=[''],
					)
				else:
					info_img = dict(
						img_path=f"{cls_name}/{data['image_path']}",
						mask_path=f"{cls_name}/{data['mask_path']}",
						cls_name=cls_name,
						specie_name=data['anomaly_class'],
						anomaly=1,
						caption=[''],
					)
				data_cls_all.append(info_img)
			meta_info[cls_name] = data_cls_all
		
		self.data_all = []
		for cls_name in self.cls_names:
			data_cls_all = meta_info[cls_name]
			self.data_all.extend(data_cls_all)

		if use_captions:
			self.captions, self.labels = get_captions(self.root, train, self.caption_folder)

			for ind, caption in enumerate(self.captions):
				assert self.data_all[ind]['img_path'] in self.labels[ind]
				self.data_all[ind]['caption'] = [caption]
		
		self.cls_to_data_d = {}
		for lbl in self.cls_names:
			self.cls_to_data_d[lbl] = []

		for dd in self.data_all:
			self.cls_to_data_d[dd['cls_name']].append(dd['img_path']) 
	
		if shuffle_data and self.train:
			random.shuffle(self.data_all)
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def __getitem__(self, index):

		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly, caption = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly'], data['caption']
		img_path = os.path.join(self.root, img_path) 

		source_filename = img_path
		target_filename = img_path

		if anomaly == 0:
			mask = Image.fromarray(np.zeros(self.image_size), mode='L')
		else:
			mask = Image.open(os.path.join(self.root, mask_path)).convert('L')

		source = Image.open(os.path.join(self.root, source_filename)).convert('RGB')
		target = Image.open(os.path.join(self.root, target_filename)).convert('RGB')
		
		source = self.transform(source)
		target = self.transform(target)
		mask = self.target_transform(mask)

		return dict(jpg=target, txt=caption[0], hint=source, mask=mask, filename=img_path, clsname=cls_name, label=int(anomaly))
