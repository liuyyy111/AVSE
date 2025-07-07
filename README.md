# Linguistic-Aware Patch Slimming Framework for Fine-grained Cross-Modal Alignment

The official codes for our paper "[Asymmetric Visual Semantic Embedding Framework for Efficient Vision-Language Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/32605)", which is accepted by the Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI’25).
We referred to the implementations of [VSE++](https://github.com/fartashf/vsepp), [GPO](https://github.com/woodfrog/vse_infty), and [LAPS](https://github.com/CrossmodalGroup/LAPS) to build up the repository. 


## Introduction
Learning visual semantic similarity is a critical challenge in bridging the gap between images and texts. However, there exist inherent variations between vision and language data, such as information density, i.e., images can contain textual information from multiple different views, which makes it difficult to compute the similarity between these two modalities accurately and efficiently. In this paper, we propose a novel framework called Asymmetric Visual Semantic Embedding (AVSE) to dynamically select features from various regions of images tailored to different textual inputs for similarity calculation.
To capture information from different views in the image, we design a radial bias sampling module to sample image patches and obtain image features from various views, Furthermore, AVSE introduces a novel module for efficient computation of visual semantic similarity between asymmetric image and text embeddings.
 Central to this module is the presumption of foundational semantic units within the embeddings, denoted as ``meta-semantic embeddings." It segments all embeddings into meta-semantic embeddings with the same dimension and calculates visual semantic similarity by finding the optimal match of meta-semantic embeddings of two modalities. 
Our proposed AVSE model is extensively evaluated on the large-scale MS-COCO and Flickr30K datasets, demonstrating its superiority over recent state-of-the-art methods.  


## Preparation

### Environments
We recommended the following dependencies:
- python >= 3.8
- torch >= 1.12.0
- torchvision >= 0.13.0
- transformers >=4.32.0
- opencv-python
- tensorboard


### Datasets

We have prepared the caption files for two datasets in  `data/` folder, hence you just need to download the images of the datasets. 
The Flickr30K (f30k) images can be downloaded in [flickr30k-images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). The MSCOCO (coco) images can be downloaded in [train2014](http://images.cocodataset.org/zips/train2014.zip), and [val2014](http://images.cocodataset.org/zips/val2014.zip).
We hope that the final data are organized as follows:


```
data
├── coco  # coco captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── testall_ids.txt
│   ├── testall_caps.txt
│   └── id_mapping.json
│
├── f30k  # f30k captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── test_ids.txt
│   ├── test_caps.txt
│   └── id_mapping.json
│
├── flickr30k-images # f30k images
│
├── coco-images # coco images
│   ├── train2014
│   └── val2014
```


## Training
First, we set up the **arguments**, detailed information about the arguments is shown in ```arguments.py```.

- `--dataset`: the chosen datasets, e.g., `f30k` and `coco`.
- `--data_path`: the root path of datasets, e.g., `data/`.
- `--multi_gpu`: whether to use the multiple GPUs (DDP) to train the models.
- `--gpu-id`, the chosen GPU number, e.g., 0-7.
- `--logger_name`, the path of logger files, e.g., `runs/f30k_test` or `runs/coco_test`


Then, we run the ```train.py``` for model training. 
The models need about 20,000 GPU-Memory (one 3090 GPU) when batch size = 64 and about 40,000 GPU-Memory (one A40 GPU) when batch size = 108.
You need to modify the batch size according to the hardware conditions, and we also support the **multiple GPUs** training. 
Besides, considering the GPU-memory limitation, we don't integrate the Gumbel-softmax sampling for the patch selection in the repository. 
The performances are not affected much but GPU-memory is reduced a lot (see more details in the paper).

```
## single GPU

### vit_224 + f30k 
python train.py --dataset f30k --gpu-id 0 --img_res 256 --logger_name runs/f30k_vit_224 --batch_size 128 --vit_type google/vit-base-patch16-224-in21k  --embed_size 1024 

### swin_224 + f30k
python train.py --dataset f30k --gpu-id 0 --img_res 256 --logger_name runs/f30k_swin_224 --batch_size 128 --vit_type microsoft/swin-base-patch4-window7-224-in22k  --embed_size 1024 

### vit_224 + coco 
python train.py --dataset coco --gpu-id 0 --img_res 256 --logger_name runs/coco_vit_224 --batch_size 128 --vit_type google/vit-base-patch16-224-in21k --embed_size 1024 

### swin_224 + coco
python train.py --dataset coco --gpu-id 0 --img_res 256 --logger_name runs/coco_swin_224 --batch_size 128 --vit_type microsoft/swin-base-patch4-window7-224-in22k  --embed_size 1024


## multiple GPUs

### vit_384 + f30k 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train.py --dataset f30k --multi_gpu 1 --img_res 384 --logger_name runs/f30k_vit_384 --batch_size 64 --vit_type vit-base-patch16-384 --embed_size 1024 

### swin_384 + f30k 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py --dataset f30k --multi_gpu 1 --img_res 384 --logger_name runs/f30k_swin_384 --batch_size 32 --vit_type microsoft/swin-base-patch4-window12-384-in22k --embed_size 1024 
```

## Evaluation
Run ```eval.py``` to evaluate the trained models on f30k or coco datasets, and you need to specify the model paths.

```
python eval.py --dataset f30k --data_path data/ --gpu-id 0
python eval.py --dataset coco --data_path data/ --gpu-id 1
```



## Reference

```
@inproceedings{liu2025asymmetric,
  title={Asymmetric Visual Semantic Embedding Framework for Efficient Vision-Language Alignment},
  author={Liu, Yang and Liu, Mengyuan and Huang, Shudong and Lv, Jiancheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={5676--5684},
  year={2025}
}
```

