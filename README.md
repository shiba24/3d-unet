# 3D UNet implementation

This repository includes Tensorflow (v1.0), PyTorch, and Chainer (v2.0)

NOTE: This is not official implementation. 

[The original paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/OB16a/oliveira16icra.pdf) is:
 Özgün Çiçek, Ahmed Abdulkadir, S. Lienkamp, Thomas Brox & Olaf Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9901, 424--432, Oct 2016


## Data preparation

Download [SLIVER07 dataset](http://sliver07.org/index.php). You need to register to download it!

Please let me know if you know better 3D voxel dataset to apply 3D convolutional neural network!


## PyTorch implementation

### Requirements

 - pytorch-0.1.11

### Training

```
cd pytorch
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



## GPU memory requirement and Result

Now in prep.

# LICENSE

MIT LICENSE.

# Author

[shiba24](https://github.com/shiba24/), April 2017.
