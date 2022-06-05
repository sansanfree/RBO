# SiamRPN++-RBO
Our code is based on [PySOT](https://github.com/STVIR/pysot) repository. You may check the original [README.md](https://github.com/STVIR/pysot/blob/master/README.md) of PySOT. 

## 1. Environment setup
This code has been tested on centos 7(Ubuntu is also OK), Python 3.6, Pytorch 1.1.0(Pytorch 1.2,1.3,1.4 and 1.5 are also OK, but for Pytorch 1.7.0 and above versions, the testing results will have slight difference), CUDA 10.0. Please install related libraries before running this code:
```bash
python setup.py build_ext --inplace
```
### Add SiamRPN++-RBO to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/SiamRPN++-RBO:$PYTHONPATH

## 2. Test
Download the pretrained model: [Google driver](https://drive.google.com/drive/folders/1BLZfzHEN4GG_29FpALKSlILXY7UyQ1Xa)  or [BaiduYun](https://pan.baidu.com/s/1a-UN4ZkjeLDGqIiF6TLZkg  code: 4oh4) and put them into `checkpoint` directory.

Download testing datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1AWMBvdFs9qg58wEdoZ5pUA code: hkfp) or [Google driver](https://drive.google.com/drive/folders/1BP7NDhMUQvrgdJSQ8MieVzLRG-mbYkTU). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.


####### GOT-10K dataset #########3##
python tools/test.py  --dataset GOT-10k  --snapshot checkpoints/SiamRPN++-RBO-got10k.pth --config experiments/test/GOT-10k/config.yaml

####### GOT-10K dataset #########3##
python tools/test.py                                \
	--dataset got10k                      \ # dataset_name
	--snapshot checkpoint/SiamRPN++-RBO-SiamRPN++-RBO.pth  \ # tracker_name

####### GOT-10K dataset #########3##
python tools/test.py                                \
	--dataset got10k                      \ # dataset_name
	--snapshot checkpoint/SiamRPN++-RBO-SiamRPN++-RBO.pth  \ # tracker_name


The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Train
### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)

Scripts to prepare training dataset are listed in `training_dataset` directory.

## If u are confused with Prepare training datasets, please refers to SiamBAN[https://github.com/hqucv/siamban] for more details about setting training dataset.


### Download pretrained backbones
Download pretrained backbones from [google driver](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) or [BaiduYun](https://pan.baidu.com/s/1pYe73PjkQx4Ph9cd3ePfCQ) (code: 5o1d) and put them into `pretrained_models` directory.

### Train a model
To train the SiamRPN++-RBO model, run `train.py` with the desired configs:

### training got10k model
cd experiments/train/got10k 
### training general model
cd experiments/train/fulldata  

CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    ../../../tools/train.py --cfg config.yaml



We use four RTX 1080TI for training.

## 4. Evaluation
We provide the raw tracking results of OTB100, VOT2016, UAV123, NFS30, GOT-10K,TC128 and LaSOT at . If you want to evaluate the tracker, please put those results into  `results` directory.

#for example, evaluation on th got10k dataset
```
python tools/eval.py --dataset GOT-10K            

```

## 5. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.


## 6. Cite
If you use RBO in your work please cite our paper:
> @InProceedings{Tang_2022_CVPR,  
   author = {Feng Tang, Qiang Ling},  
   title = {Ranking-Based Siamese Visual Tracking},  
   booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
   month = {June},  
   year = {2022}  
}

















## Introduction

The goal of PySOT is to provide a high-quality, high-performance codebase for visual tracking *research*. It is designed to be flexible in order to support rapid implementation and evaluation of novel research. PySOT includes implementations of the following visual tracking algorithms:

- [SiamMask](https://arxiv.org/abs/1812.05050)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- [SiamFC](https://arxiv.org/abs/1606.09549)

using the following backbone network architectures:

- [ResNet{18, 34, 50}](https://arxiv.org/abs/1512.03385)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](#references) below.

Evaluation toolkit can support the following datasets:

:paperclip: [OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf) 
:paperclip: [VOT16/18/19](http://votchallenge.net) 
:paperclip: [VOT18-LT](http://votchallenge.net/vot2018/index.html) 
:paperclip: [LaSOT](https://arxiv.org/pdf/1809.07845.pdf) 
:paperclip: [UAV123](https://arxiv.org/pdf/1804.00518.pdf)

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [PySOT Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for PyTorch and PySOT in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using PySOT

### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

### Download models
Download models in [PySOT Model Zoo](MODEL_ZOO.md) and put the model.pth in the correct directory in experiments

### Webcam demo
```bash
python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets
Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker
```bash
cd experiments/siamrpn_r50_l234_dwxcorr
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```
The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker
assume still in experiments/siamrpn_r50_l234_dwxcorr_8gpu
``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

###  Training :wrench:
See [TRAIN.md](TRAIN.md) for detailed instruction.


### Getting Help :hammer:
If you meet problem, try searching our GitHub issues first. We intend the issues page to be a forum in which the community collectively troubleshoots problems. But please do **not** post **duplicate** issues. If you have similar issue that has been closed, you can reopen it.

- `ModuleNotFoundError: No module named 'pysot'`

:dart:Solution: Run `export PYTHONPATH=path/to/pysot` first before you run the code.

- `ImportError: cannot import name region`

:dart:Solution: Build `region` by `python setup.py build_ext —-inplace` as decribled in [INSTALL.md](INSTALL.md).


## References

- [Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/abs/1812.05050).
  Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://arxiv.org/abs/1812.11703).
  Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [Distractor-aware Siamese Networks for Visual Object Tracking](https://arxiv.org/abs/1808.06048).
  Zheng Zhu, Qiang Wang, Bo Li, Wu Wei, Junjie Yan, Weiming Hu.
  The European Conference on Computer Vision (ECCV), 2018.

- [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html).
  Bo Li, Wei Wu, Zheng Zhu, Junjie Yan, Xiaolin Hu.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

- [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549).
  Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H. S. Torr.
  The European Conference on Computer Vision (ECCV) Workshops, 2016.
  
## Contributors

- [Fangyi Zhang](https://github.com/StrangerZhang)
- [Qiang Wang](http://www.robots.ox.ac.uk/~qwang/)
- [Bo Li](http://bo-li.info/)

## License

PySOT is released under the [Apache 2.0 license](https://github.com/STVIR/pysot/blob/master/LICENSE). 
