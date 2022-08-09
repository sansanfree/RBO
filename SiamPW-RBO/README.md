# SiamPW-RBO
Our code is based on [SiamBAN](https://github.com/hqucv/siamban) repository. Thansk for their contributions.

## 1. Environment setup
This code has been tested on centos 7(Ubuntu is also OK), Python 3.6, Pytorch 1.1.0(Pytorch 1.2,1.3,1.4 and 1.5 are also OK, but for Pytorch 1.7.0 and above versions, the testing results will have slight difference), CUDA 10.0. Please install related libraries before running this code:

python setup.py build_ext --inplace

### Add SiamPW-RBO to your PYTHONPATH
#export PYTHONPATH=/path/to/SiamPW-RBO:$PYTHONPATH

## 2. Test
Download the pretrained model: [Google driver](https://drive.google.com/drive/folders/1VbcFz33dsJwp8SZlZUEb6tTUjDIDWFrk)  and put them into `checkpoints` directory.

Download testing datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1AWMBvdFs9qg58wEdoZ5pUA)(code: hkfp) or [Google driver](https://drive.google.com/drive/folders/1BP7NDhMUQvrgdJSQ8MieVzLRG-mbYkTU). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.


####### GOT-10K dataset ###########

python tools/test.py  --dataset GOT-10k  --snapshot checkpoints/SiamPW-RBO-got10k.pth --config experiments/test/GOT-10k/config.yaml

####### OTB100 dataset ###########
python tools/test.py   --dataset OTB100  --snapshot checkpoints/SiamPW-RBO-general-OT.pth  --config experiments/test/OTB100/config.yaml

####### TC128 dataset ###########
python tools/test.py   --dataset TC128  --snapshot checkpoints/SiamPW-RBO-general-OT.pth  --config experiments/test/TC128/config.yaml

####### NFS30 dataset ###########
python tools/test.py   --dataset NFS30 --snapshot checkpoints/SiamPW-RBO-general-LUVN.pth  --config experiments/test/NFS30/config.yaml

####### VOT2016 dataset ###########
python tools/test.py   --dataset VOT2016 --snapshot checkpoints/SiamPW-RBO-general-LUVN.pth  --config experiments/test/VOT2016/config.yaml

####### UAV123 dataset ###########
python tools/test.py   --dataset UAV123 --snapshot checkpoints/SiamPW-RBO-general-LUVN.pth  --config experiments/test/UAV123/config.yaml

####### LaSOT dataset ###########
python tools/test.py   --dataset LaSOT --snapshot checkpoints/SiamPW-RBO-general-LUVN.pth  --config experiments/test/LaSOT/config.yaml

The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Train
### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)
* [LASOT](https://cis.temple.edu/lasot/)

Scripts to prepare training dataset are listed in `training_dataset` directory.

 If you are confused with preparing training datasets, please refers to SiamBAN[https://github.com/hqucv/siamban] for more details about setting training dataset.


### Download pretrained backbones
Download pretrained backbones from [google driver](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) or [BaiduYun](https://pan.baidu.com/s/1pYe73PjkQx4Ph9cd3ePfCQ) (code: 5o1d) and put them into `pretrained_models` directory.

### Train a model
To train the SiamPW-RBO model, run `train.py` with the desired configs:

### training got10k model
cd experiments/train/got10k 
### training general model
cd experiments/train/fulldata  

CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    ../../../tools/train.py --cfg config.yaml



We use four RTX 1080ti for training.

## 4. Evaluation
We provide the raw tracking results of OTB100, VOT2016, UAV123, NFS30, GOT-10K,TC128 and LaSOT at . If you want to evaluate the tracker, please put those results into  `results` directory.

##for example, evalution on the OTB100 dataset

python tools/eval.py --dataset OTB100            


## 5. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot) and [siamban](https://github.com/hqucv/siamban). We would like to express our sincere thanks to the contributors.


## 6. Cite
If you use RBO in your work please cite our paper:
> @InProceedings{tang_2022_CVPR,  
   author = {Feng Tang, Qiang Ling},  
   title = {Ranking-Based Siamese Visual Tracking},  
   booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
   month = {June},  
   year = {2022}  
}


