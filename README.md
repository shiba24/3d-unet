# 3D UNet implementation [WIP]  [Contribution Welcome]

This repository includes Tensorflow (v1.0), PyTorch, and Chainer (v2.0) implementations of 3D UNet, semantic segmentation neural network for 3D voxel data.

NOTE: This is not official implementation. Currently only Chainer implementation works well.


[The original paper](https://arxiv.org/abs/1606.06650) is:
 Özgün Çiçek, Ahmed Abdulkadir, S. Lienkamp, Thomas Brox & Olaf Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9901, 424--432, Oct 2016


## Data preparation

Download [SLIVER07 dataset](http://sliver07.org/index.php). You need to register to download it.

Please let me know if you know better 3D voxel dataset to apply 3D convolutional neural network!


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


# LICENSE

MIT LICENSE.

# Author

[shiba24](https://github.com/shiba24/), April 2017.
