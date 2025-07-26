
# 📚 VLMDiff: Leveraging Vision-Language Models for Multi-Class Anomaly Detection with Diffusion
<!-- [Conference/Journal Name], [Year]   -->

[Samet Hicsonmez<sup>#</sup>](https://scholar.google.com/citations?user=biHfDhUAAAAJ&hl),
[Abd El Rahman Shabayek<sup>#</sup>](https://scholar.google.com/citations?user=185kRdEAAAAJ),
[Djamila Aouada<sup>#</sup>](https://scholar.google.com/citations?user=WBmJVSkAAAAJ)

[<sup>#</sup>Interdisciplinary Centre for Security, Reliability, and Trust (SnT), University of Luxembourg](https://www.uni.lu/snt-en/research-groups/cvi2/), 

[![arXiv](https://img.shields.io/badge/arXiv-PDF-red)](link_to_arxiv) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)



## 🧠 Abstract
Detecting visual anomalies in diverse, multi-class real-world images is a significant challenge. We introduce VLMDiff, a novel unsupervised multi-class visual anomaly detection
framework. It integrates a Latent Diffusion Model (LDM)
with a Vision-Language Model (VLM) for enhanced anomaly
localization and detection. Specifically, a pre-trained VLM
with a simple prompt extracts detailed image descriptions,
serving as additional conditioning for LDM training. Current
diffusion-based methods rely on synthetic noise generation,
limiting their generalization and requiring per-class model
training, which hinders scalability. VLMDiff, however, lever ages VLMs to obtain normal captions without manual annotations or additional training. These descriptions condition
the diffusion model, learning a robust normal image feature
representation for multi-class anomaly detection. Our method
achieves competitive performance, improving the pixel-level
Per-Region-Overlap (PRO) metric by up to 25 points on the
Real-IAD dataset and 8 points on the COCO-AD dataset, out performing state-of-the-art diffusion-based approaches.


## 🖥️ Quick Start

### 1. Clone and Install


First create a new conda environment, and install all required packages.

```
git clone https://gitlab.com/uniluxembourg/snt/cvi2/open/space/vlmdiff
cd vlmdiff

conda create -n vlmdiff python=3.10
conda activate vlmdiff
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


### 2.Dataset

### 2.1 Real-IAD
- **Create the Real-IAD dataset directory**. Download the Real-IAD dataset from [Real-IAD](https://realiad4ad.github.io/Real-IAD/). The Real-IAD dataset directory should be as follows. 

```
|-- /path/to/data/dir
    |-- real_iad
        |-- audiojack
            |-- NG
                |-- BX
                    |-- S0001
                |-- HS
                    |-- S0001
            |-- OK
                |-- S0001
        |-- bottle_cap
            |-- NG
                |-- AK
                    |-- S0001
                |-- HS
                    |-- S0001
            |-- OK
                |-- S0001
    |-- real_iad_jsons
        |-- audiojack.json
        |-- bottle_cap.json

```

### 2.2 COCO-AD
- **Create the COCO-AD dataset directory**. Download the COCO (2017) dataset from [COCO](https://cocodataset.org/#download). Thn prepare the COCO-AD dataset using the scripts in [ADER](https://github.com/zhangzjn/ADer/blob/main/data/gen_benchmark/coco.py). The COCO-AD dataset directory should be as follows.  
```
|-- /path/to/data/dir
    |-- COCO
        |-- annotations
        |-- train2017
        |-- val2017
        |-- val2017_mask_ad_20_0
        |-- val2017_mask_ad_20_1
        |-- val2017_mask_ad_20_2
        |-- val2017_mask_ad_20_3
        |-- meta_20_0.json
        |-- meta_20_1.json
        |-- meta_20_2.json
        |-- meta_20_3.json

```

### 2.3 MVTec-AD
- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). The MVTec-AD dataset directory should be as follows. 

```
|-- /path/to/data/dir
    |-- mvtec_anomaly_detection
        |-- bottle
            |-- ground_truth
                |-- broken_large
                    |-- 000_mask.png
                |-- broken_small
                    |-- 000_mask.png
                |-- contamination
                    |-- 000_mask.png
            |-- test
                |-- broken_large
                    |-- 000.png
                |-- broken_small
                    |-- 000.png
                |-- contamination
                    |-- 000.png
                |-- good
                    |-- 000.png
            |-- train
                |-- good
                    |-- 000.png
```

### 2.4 VisA
- **Create the VisA dataset directory**. Download the VisA dataset from [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). The VisA dataset directory should be as follows. 

```
|-- /path/to/data/dir
    |-- visa
        |-- candle
            |-- Data
                |-- Images
                    |-- Anomaly
                        |-- 000.JPG
                    |-- Normal
                        |-- 0000.JPG
                |-- Masks
                    |--Anomaly 
                        |-- 000.png        
```
---

## 3. Finetune the Autoencoders
- Finetune the Autoencoders first by downloading the pretrained Autoencoders from [kl-f8.zip](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip). Move it to `./models/autoencoders.ckpt`.
And finetune the model with running

```
# Real-IAD
python finetune_autoencoder.py --resume_path ./models/autoencoders.ckpt --coco_part -1 --data_set real_iad --data_path /path/to/data/dir/real_iad

# COCO part 0
python finetune_autoencoder.py --resume_path ./models/autoencoders.ckpt --coco_part 0 --data_set coco --data_path /path/to/data/dir/COCO

# MVTEC-AD
python finetune_autoencoder.py --resume_path ./models/autoencoders.ckpt --coco_part -1 --data_set mvtec --data_path /path/to/data/dir/mvtec_anomaly_detection

# VISA
python finetune_autoencoder.py --resume_path ./models/autoencoders.ckpt --coco_part -1 --data_set visa --data_path /path/to/data/dir/visa
```


- Once finished the finetuned model is under the folder `./lightning_logs/version_x/checkpoints/epoch=xxx-step=xxx.ckpt`.
Then move it to the folder with changed name `./models/mvtec_ae.ckpt`.
- If you use the given pretrained autoencoder model, you can go step 4 to build the model.

| Autoencoder        | Pretrained Model                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------|
| Real-IAD First Stage Autoencoder | [real_iad_fs]() |
| COCO part0 First Stage Autoencoder | [coco_part0_fs]() |
| COCO part1 First Stage Autoencoder | [coco_part1_fs]() |
| COCO part2 First Stage Autoencoder | [coco_part2_fs]() |
| COCO part3 First Stage Autoencoder | [coco_part3_fs]() |
| MVTec First Stage Autoencoder | [mvtecad_fs](https://drive.google.com/file/d/1vDfywjGqoWRHMxj-5fifujK29_XyHuCQ/view?usp=sharing) |
| VisA First Stage Autoencoder  | [visa_fs](https://drive.google.com/file/d/1zycpAbWwIVodwTo0Bh1oK8xKliuTT3ul/view?usp=sharing)    |


## 4. Generate Image Descriptions

We have two scripts for the description extraction using different VLMs.

- DeepseekVL is used from [LMDeploy](https://github.com/InternLM/lmdeploy) repository. You may need to create a separate environment for it, as it may create problems with the installed packages. 

```
# Real-IAD, or MVTec-AD or VISA. just change the data_set and data_path
python captioner_lmdeploy_deepseekvl.py --coco_part -1 --data_set real_iad --data_path /path/to/data/dir/real_iad --vlm_model deepseekvl --prompt_type pd

# COCO-AD, no need to run for each split
python captioner_lmdeploy_deepseekvl.py --coco_part 0 --data_set coco --data_path /path/to/data/dir/COCO --vlm_model deepseekvl --prompt_type pd
# For COCO: after it is done format it using, first edit the contents 
python utils/coco_format_converter.py 
```

- InternVL and BLIP is used from [LAVIS](https://github.com/salesforce/LAVIS) repository. You may need to create a separate environment for it, as it may create problems with the installed packages. 

```
# Real-IAD, or MVTec-AD or VISA. just change the data_set and data_path
python captioner.py --coco_part -1 --data_set real_iad --data_path /path/to/data/dir/real_iad --vlm_model internvl --prompt_type pd

# BLIP2 descriptions
python captioner.py --coco_part -1 --data_set real_iad --data_path /path/to/data/dir/real_iad --vlm_model blip --prompt_type pd

# COCO-AD, no need to run for each split
python captioner.py --coco_part 0 --data_set coco --data_path /path/to/data/dir/COCO --vlm_model internvl --prompt_type pd
# For COCO: after it is done format it using, first edit the contents. This puts the caption json to coco/annotations folder.
python utils/coco_format_converter.py 
```

- In the end you should move the caption folders to the same folder as datasets. For instance, the new Real-IAD folder should looks like:

```
|-- /path/to/data/dir
    |-- real_iad
        |-- audiojack
            |-- NG
                |-- BX
                    |-- S0001
                |-- HS
                    |-- S0001
            |-- OK
                |-- S0001
        |-- bottle_cap
            |-- NG
                |-- AK
                    |-- S0001
                |-- HS
                    |-- S0001
            |-- OK
                |-- S0001
    |-- real_iad_jsons
        |-- audiojack.json
        |-- bottle_cap.json
    |-- realiad_captions_internvl_pd
        |-- test_captions_realiad_internvl.json
        |-- test_labels_realiad_internvl.json
        |-- train_captions_realiad_internvl.json
        |-- train_labels_realiad_internvl.json
    |-- realiad_captions_deepseekvl_pd
    |-- realiad_captions_blip_pd

```


## 5. Prepare the Model
- We use the pre-trianed stable diffusion v1.5, the finetuned autoencoders and the Semantic-Guided Network to build the full needed model for training.
The stable diffusion v1.5 could be downloaded from ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). Move it under the folder `./models/v1-5-pruned.ckpt`. 
Then run the code to get the output model `./models/vlmdiff.ckpt`.

```
python build_model.py
```


## 6. Train
- Training the model by running

```
# REAL-IAD. MVTEC-AD and VISA are similar to train
python train.py --resume_path ./models/vlmdiff.ckpt --coco_part -1 --data_set realiad --data_path /path/to/data/dir/real_iad --exp_name realiad_internvl --use_captions  --caption_folder realiad_captions_internvl_pd

# COCO-AD
python train.py --resume_path ./models/vlmdiff.ckpt --coco_part 0 --data_set coco --data_path /path/to/data/dir/coco --exp_name coco_internvl --use_captions --caption_model _internvl

```
- Batch size, learning rate and gpus could be edited in `train.py`.


## 7. Test
We train our models for **100** epochs on Real-IAD and COCO, and **300** epochs on MVTec-AD and VISA datasets.
The output of the saved checkpoint could be found under `./ckpt_{exp_name}/model_epoch=XXX.ckpt`. 

For evaluation and visualization, run the following:

```
# Real-IAD, MVTEC and VISA. No text guidance during inference.
python test.py --resume_path ./ckpt_{exp_name}/model_epoch=099.ckpt --coco_part 0 --data_set realiad --data_path /path/to/data/dir/real_iad --exp_name realiad_internvl --use_dino --dino_version v1s8 --start_ind 0 --end_ind 120000 # --save_visuals


# COCO: we are using text guidance for testing as well.
python test.py --resume_path ./ckpt_{exp_name}/model_epoch=099.ckpt --coco_part 0 --data_set realiad --data_path /path/to/data/dir/coco --exp_name coco_part0_internvl --use_captions --caption_model _internvl # --save_visuals

```

You can explore different commandline arguments in the `test.py`.

The images are saved under `./log_image/, where
- `xxx-input.jpg` is the input image.
- `xxx-reconstruction.jpg` is the reconstructed image through autoencoder without diffusion model.
- `xxx-features.jpg` is the feature map of the anomaly score.
- `xxx-samples.jpg` is the reconstructed image through the autoencoder and diffusion model.
- `xxx-heatmap.png` is the heatmap of the anomaly score.


---

## 📊 Results
|   Dataset    | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | mAU-PRO<sub>R</sub> |
|:-----------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|
|  Real-IAD   |        78.0 | 73.2 | 69.8          |        97.1 | 25.5 | 32.3          |        87.7         |
|   COCO-AD   |        59.1 | 51.2 | 62.9          |        69.0 | 15.3 | 22.4          |        38.8         |
|  MVTec-AD   |        90.6 | 95.7 | 94.2          |        95.9 | 51.8 | 53.6          |        89.4         |
|    VisA     |        80.9 | 83.9 | 80.7          |        97.0 | 28.3 | 33.6          |        81.0         |

---

<!-- ## 📁 Folder Structure

```
.
├── data/              # Datasets
├── ddim_inversion.py  # Generate reconstructions
├── test.py            # Evaluate the performance
└── README.md
```

--- -->

## 📜 Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{your2025paper,
  title={Your Paper Title},
  author={Your, Name and Coauthor, Name},
  booktitle={Proceedings of Conference},
  year={2025}
}
```

---

## 🪪 License

This repository is licensed under the Apache License. See the [`LICENSE`](LICENSE) file for details.

---

## 🙏 Acknowledgements
This repo builds upon open-source contributions from:

 * [DiAD](https://github.com/lewandofskee/DiAD)
 * [LDM](https://github.com/CompVis/latent-diffusion) 
 * [ControlNet](https://github.com/lllyasviel/ControlNet)
