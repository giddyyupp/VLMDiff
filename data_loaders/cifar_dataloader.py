import os
import json
import random
import pickle
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from data_loaders.data_utils import data_transforms, gt_transforms


class CifarAD(Dataset):
	def __init__(self, root, train=True, use_captions=False, cifar_type='cifar100', data_tf=None):
		
		# ---------- for one-class classification ----------
		self.type_cifar = cifar_type
		self.cls_names = ['cifar']
		self.uni_setting = True
		self.one_cls_train = True
		self.split_idx = 0
		
		self.root = root
		self.train = train
		self.image_size = (256, 256)
		self.transform = data_transforms(self.image_size) if data_tf is None else data_tf
		self.target_transform = gt_transforms(self.image_size)
		self.use_captions = use_captions
		self.image_names_all = []
		
		if self.use_captions:
			with open(os.path.join(self.root, f'{self.type_cifar}_captions/train_captions.json')) as ff:
				self.captions = json.load(ff)
			with open(os.path.join(self.root, f'{self.type_cifar}_captions/test_captions.json')) as ff:
				self.captions_test = json.load(ff)

			self.captions.extend(self.captions_test)

		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		# init splits
		if self.type_cifar == 'cifar10':
			self.root = f'{self.root}/cifar-10-batches-py'
			train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
			test_list = ['test_batch']
			cate_num = 10
		else:
			self.root = f'{self.root}/cifar-100-python'
			train_list = ['train']
			test_list = ['test']
			cate_num = 100
		cate_num_half = cate_num // 2
		if self.uni_setting:
			self.splits = [{'train': self.range(0, cate_num_half, 1), 'test': self.range(cate_num_half, cate_num, 1)},
							{'train': self.range(cate_num_half, cate_num, 1), 'test': self.range(0, cate_num_half, 1)},
							{'train': self.range(0, cate_num, 2), 'test': self.range(1, cate_num, 2)},
							{'train': self.range(1, cate_num, 2), 'test': self.range(0, cate_num, 2)}, ]
		else:
			self.splits = []
			for idx in range(cate_num):
				cates = self.range(0, cate_num, 1)
				cates.remove(idx)
				if self.one_cls_train:
					self.splits.append({'train': [idx], 'test': cates})
				else:
					self.splits.append({'train': cates, 'test': [idx]})
		splits = self.splits[self.split_idx]
		# load data
		imgs, pseudo_cls_names, phases = [], [], []
		for idx, data_list in enumerate([train_list, test_list]):
			for file_name in data_list:
				file_path = f'{self.root}/{file_name}'
				with open(file_path, 'rb') as f:
					entry = pickle.load(f, encoding='latin1')
					imgs.append(entry['data'])
					self.image_names_all.extend(entry['filenames'])
					pseudo_cls_names.extend(entry['labels'] if 'labels' in entry else entry['fine_labels'])
					phases.extend([idx] * len(entry['data']))
		self.imgs = np.vstack(imgs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
		self.pseudo_cls_names = np.array(pseudo_cls_names)
		self.phases = np.array(phases)  # phase 0-> train, 1-> test
		# assign data
		self.data_all = []
		if self.train:
			for idx in range(len(self.imgs)):
				img, cls_name, phase, image_name = self.imgs[idx], self.pseudo_cls_names[idx], self.phases[idx], self.image_names_all[idx]
				caption = self.captions[idx] if self.use_captions else [""]
				if cls_name in splits['train'] and phase == 0:
					self.data_all.append([img, self.cls_names[0], 0, image_name, caption])
		else:
			if self.one_cls_train and self.type_cifar == 'cifar10':
				cls_cnt, max_cnt = [0] * 10, [111] * 10
				for cls_name in splits['train']:
					max_cnt[cls_name] = 1000
					max_cnt[(cls_name + 1) % 10] = 112
			for idx in range(len(self.imgs)):
				img, cls_name, phase, image_name = self.imgs[idx], self.pseudo_cls_names[idx], self.phases[idx], self.image_names_all[idx]
				caption = self.captions[idx] if self.use_captions else [""]
				if self.uni_setting:
					if phase == 1:
						self.data_all.append([img, self.cls_names[0], 0 if cls_name in splits['train'] else 1, image_name, caption])
				else:
					if self.one_cls_train and self.type_cifar == 'cifar10':
						if phase == 1:
							if cls_cnt[cls_name] < max_cnt[cls_name]:
								cls_cnt[cls_name] += 1
								self.data_all.append([img, self.cls_names[0], 0 if cls_name in splits['train'] else 1, image_name, caption])
					else:
						if phase == 1:
							self.data_all.append([img, self.cls_names[0], 0 if cls_name in splits['train'] else 1, image_name, caption])
						elif phase == 0:
							if cls_name in splits['test']:
								self.data_all.append([img, self.cls_names[0], 1, image_name, caption])
		# random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

	@staticmethod
	def range(start, stop, step):
		return list(range(start, stop, step))

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		img, cls_name, anomaly, image_name, caption = self.data_all[index]
		source = Image.fromarray(img)
		target = Image.fromarray(img)
		mask = Image.fromarray(np.zeros((source.size[0], source.size[1])) if anomaly == 0 else np.ones((source.size[0], source.size[1])), mode='L')
		# img = self.transform(img) if self.transform is not None else img
		# img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		# img_mask = [] if img_mask is None else img_mask

		source = self.transform(source)
		target = self.transform(target)
		mask = self.target_transform(mask)

		return dict(jpg=target, txt=caption[0], hint=source, mask=mask, filename=image_name, clsname=cls_name, label=int(anomaly))


if __name__ == '__main__':
	from argparse import Namespace as _Namespace

	cfg = _Namespace()
	data = _Namespace()
	data.sampler = 'naive'
	# ========== MVTec ==========
	# data.root = 'data/mvtec'
	# data.meta = 'meta.json'
	# # data.cls_names = ['bottle']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/mvtec3d'
	# data.meta = 'meta.json'
	# # data.cls_names = ['bagel']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/coco'
	# data.meta = 'meta_20_0.json'
	# data.cls_names = ['coco']
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/visa'
	# data.meta = 'meta.json'
	# # data.cls_names = ['candle']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# ========== Cifar ==========
	data.type = 'DefaultAD'
	# data.root = '/mnt/isilon/projects/sigcom_cv/shicsonmez/ad/data/'
	data.root = '/mnt/isilon/shicsonmez/ad/data'
	data.type_cifar = 'cifar10'
	data.cls_names = ['airplane']
	data.uni_setting = True
	data.one_cls_train = True
	data.split_idx = 0
	data_fun = CifarAD

	# ========== Tiny ImageNet ==========
	# data.root = 'data/tiny-imagenet-200'
	# data.cls_names = ['tin']
	# data.loader_type = 'pil'
	# data.split_idx = 0
	# data_fun = TinyINAD

	# ========== Real-IAD ==========
	# data.root = 'data/realiad/explicit_full'
	# # data.cls_names = ['audiojack']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data.views = ['C1', 'C2']
	# # data.views = []
	# data.use_sample = True
	# data_fun = RealIAD


	cfg.data = data
	# data_debug = data_fun(root=data.root, train=True, use_captions=True)
	data_debug = data_fun(root=data.root, train=False, use_captions=True)
	for idx, data in enumerate(data_debug):
		# break
		if idx > 1000:
			break
		print()