# Superpixel-Cost-Volume-Excitation-for-Stereo-Matching
This is the implementation of the paper Superpixel Cost Volume Excitation for Stereo Matching, PRCV 24, Shanglong Liu, Lin Qi, Junyu Dong, Wenxiang Gu, Liyi Xu[\[Arxiv\]](https://arxiv.org/abs/2411.13105)
Please contact [Shanglong Liu](https://github.com/lsl-air) (ouc3251@gmail.com) if you have any questions.

## Environment
* NVIDIA RTX 3090
* python 3.7.13
* pytorch 1.12.0

To ensure connectivity during superpixel visualization, similar to SpixelFCN , we make use of the component connection method in [SSN](https://github.com/NVlabs/ssn_superpixels) to enforce the connectivity in superpixels. The code has been included in ```/models/cython```. To compile it:
```
cd models/cython/
python setup.py install --user
cd ../..
```

## Pretrained Models
[SpixelFCN](https://pan.baidu.com/s/1GW5-U92IDiwEFItZNfoo5w?pwd=65kd)
[GwcNet-Sceneflow](https://pan.baidu.com/s/1pEUM0c-xZdQoAr3yBSiloQ?pwd=aqa3)

## Scene Flow Datasets
**Training**
Before starting joint training, load the weights of the sub-network. The pre-trained weights should be available in the following path ```/pretrained/spixel_16/SpixelNet_bsd_ckpt.tar```.

run the script `./scripts/sceneflow.sh` to train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.

**Testing**
If only the training head is used, the sub-network can be omitted during the inference stage to maintain the same parameters and computational cost as the baseline network.

## Citation
If you find our work useful in your research, please consider citing our paper:

```bibtex
@InProceedings{10.1007/978-981-97-8508-7_2,
author="Liu, Shanglong
and Qi, Lin
and Dong, Junyu
and Gu, Wenxiang
and Xu, Liyi",
title="Superpixel Cost Volume Excitation for Stereo Matching",
booktitle="Pattern Recognition and Computer Vision",
year="2025",
pages="18--31",
}
```

## Acknowledgements
This project is based on [GwcNet](https://github.com/xy-guo/GwcNet), [CoEx](https://github.com/antabangun/coex), and [SpixelFCN](https://github.com/fuy34/superpixel_fcn).