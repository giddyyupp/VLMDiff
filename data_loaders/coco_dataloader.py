import os
import json
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from data_loaders.data_utils import data_transforms, gt_transforms


class DefaultAD(Dataset):
	def __init__(self, root, meta='', train=True, use_captions=True, caption_model='', data_tf=None, shuffle_data=True):

		self.meta = meta
		self.cls_names = ['coco']

		self.root = root
		self.train = train
		self.caption_model = caption_model  # for gt captions use empty

		self.image_size = (256, 256)
		self.transform = data_transforms(self.image_size) if data_tf is None else data_tf
		self.target_transform = gt_transforms(self.image_size)

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/{self.meta}', 'r'))
		meta_info = meta_info['train' if self.train else 'test']
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names

		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])

		self.ind_to_imname = {}
		for ind, dd in enumerate(self.data_all):
			img_id = os.path.split(dd['img_path'])[-1][:-4]
			self.ind_to_imname[img_id] = ind
			self.data_all[ind].update({"caption": ""})

		if use_captions:
			# gt captions
			captions_json = f'captions_train2017{self.caption_model}.json' if self.train else f'captions_val2017{self.caption_model}.json'
			captions = json.load(open(os.path.join(self.root, 'annotations', captions_json), 'r'))
			for annot in captions['annotations']:
				im_name_padded = str(annot['image_id']).zfill(12)
				if im_name_padded in self.ind_to_imname:
					img_data = self.data_all[self.ind_to_imname[im_name_padded]]
					if 'caption' not in img_data:
						if self.caption_model:
							img_data['caption'] = annot['caption']
						else:
							img_data['caption'] = ''
					else:
						img_data['caption'] += annot['caption']					

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

		return dict(jpg=target, txt=caption, hint=source, mask=mask, filename=img_path, clsname=cls_name, label=int(anomaly))


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

	data.root = '/mnt/isilon/shared_datasets/coco'
	data.meta = 'meta_20_0.json'
	data.cls_names = ['coco']
	data_fun = DefaultAD

	# data.root = 'data/visa'
	# data.meta = 'meta.json'
	# # data.cls_names = ['candle']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# ========== Cifar ==========
	# data.type = 'DefaultAD'
	# # data.root = '/mnt/isilon/projects/sigcom_cv/shicsonmez/ad/data/'
	# data.root = '/mnt/isilon/shicsonmez/ad/data'
	# data.type_cifar = 'cifar10'
	# data.cls_names = ['airplane']
	# data.uni_setting = True
	# data.one_cls_train = True
	# data.split_idx = 0
	# data_fun = CifarAD

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

	SPLITS = []
	ano_num = 20
	for group in range(80 // ano_num):
		labels = list(range(0, 80, 1))
		labels_rm = list(range(group * ano_num, (group + 1) * ano_num))
		for label_rm in labels_rm:
			labels.remove(label_rm)
		SPLITS.append(dict(train=labels, test=labels_rm))

	# data_debug = data_fun(root=data.root, train=True, use_captions=True)
	data_debug = data_fun(data.root, data.meta, train=True, use_captions=True, caption_model='_blip', data_tf=None, shuffle_data=False)
	for idx, data in enumerate(data_debug):
		# break
		if idx > 1000:
			break
		print()
