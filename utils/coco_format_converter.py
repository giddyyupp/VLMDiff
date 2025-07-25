import os
import json

split = 'train'
method = 'deepseekvl'
json_path = f'/path/to/vlmdiff/coco_captions_{method}/{split}_captions_coco_{method}.json'
labels_path = f'/path/to/vlmdiff/coco_captions_{method}/{split}_labels_coco_{method}.json'
save_path = f'/path/to/coco/annotations/captions_{split}2017_{method}.json'


main_dict = {}
annots = []
# imgs = []

# start conversion!
captions = json.load(open(json_path, 'r'))
image_names = json.load(open(labels_path, 'r'))

for i, caption in enumerate(captions):
    image_id = image_names[i][0]
    annots.append({'id': i,
                   'image_id': image_id,
                   'caption': caption})
    
main_dict['annotations'] = annots
# main_dict['info'] = dataset_train.coco.dataset['info']
# main_dict['licenses'] = dataset_train.coco.dataset['licenses']

# dump to json
with open(save_path, 'w') as fp:
    json.dump(main_dict, fp)