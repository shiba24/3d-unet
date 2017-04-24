# 3D UNet implementation

NOTE: This is not official implementation. [Original paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/OB16a/oliveira16icra.pdf)

 Özgün Çiçek, Ahmed Abdulkadir, S. Lienkamp, Thomas Brox & Olaf Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9901, 424--432, Oct 2016


## Data preparation

```
bash prepare.sh
```


## Chainer implementation


### Requirements

- Python 2.7.11+
  - [Chainer 1.21.0.1+](https://github.com/pfnet/chainer)
  - numpy 1.9+
  - scipy 0.16+
  - six
  - matplotlib
  - tqdm
  - cv2 (opencv)

### Training


```
cd src
python 
```


## PyTorch implementation

### Requirements

 - pytorch-0.1.11


### Training

```
cd src
python
```


## GPU memory requirement


## Result

Now in prep.

# Visualize Prediction

```
python visualize.py -f PATH_TO_IMAGE_FILE
```

# LICENSE

MIT LICENSE.

# Author

[shiba24](https://github.com/shiba24/), April 2017.
