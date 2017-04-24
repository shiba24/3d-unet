import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal as w


class UNet3D(chainer.Chain):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        super(UNet3D, self).__init__(
            c0=L.ConvolutionND(3, self.in_channel, 32, 3, 1, 1, initial_bias=None),
            c1=L.ConvolutionND(3, 32, 64, 3, 1, 1, initial_bias=None),

            c2=L.ConvolutionND(3, 64, 64, 3, 1, 1, initial_bias=None),
            c3=L.ConvolutionND(3, 64, 128, 3, 1, 1, initial_bias=None),

            c4=L.ConvolutionND(3, 128, 128, 3, 1, 1, initial_bias=None),
            c5=L.ConvolutionND(3, 128, 256, 3, 1, 1, initial_bias=None),

            c6=L.ConvolutionND(3, 256, 256, 3, 1, 1, initial_bias=None),
            c7=L.ConvolutionND(3, 256, 512, 3, 1, 1, initial_bias=None),

            dc9=L.DeconvolutionND(3, 512, 512, 2, 2, initial_bias=None),
            dc8=L.ConvolutionND(3, 256 + 512, 256, 3, 1, 1, initial_bias=None),
            dc7=L.ConvolutionND(3, 256, 256, 3, 1, 1, initial_bias=None),

            dc6=L.DeconvolutionND(3, 256, 256, 2, 2, initial_bias=None),
            dc5=L.ConvolutionND(3, 128 + 256, 128, 3, 1, 1, initial_bias=None),
            dc4=L.ConvolutionND(3, 128, 128, 3, 1, 1, initial_bias=None),

            dc3=L.DeconvolutionND(3, 128, 128, 2, 2, initial_bias=None),
            dc2=L.ConvolutionND(3, 64 + 128, 64, 3, 1, 1, initial_bias=None),
            dc1=L.ConvolutionND(3, 64, 64, 3, 1, 1, initial_bias=None),

            dc0=L.ConvolutionND(3, 64, n_classes, 1, 1, initial_bias=None),

        )
        self.train = True

    def __call__(self, x, use_cudnn=False):
        test = not self.train
        e0 = F.relu(self.c0(x), use_cudnn)
        syn0 = F.relu(self.c1(e0), use_cudnn)    
        del e0

        e1 = F.max_pooling_nd(syn0, 2, 2)
        e2 = F.relu(self.c2(e1), use_cudnn)
        syn1 = F.relu(self.c3(e2), use_cudnn)
        del e1, e2

        e3 = F.max_pooling_nd(syn1, 2, 2)
        e4 = F.relu(self.c4(e3), use_cudnn)
        syn2 = F.relu(self.c5(e4), use_cudnn)
        del e3, e4
        
        e5 = F.max_pooling_nd(syn2, 2, 2)
        e6 = F.relu(self.c6(e5), use_cudnn)
        e7 = F.relu(self.c7(e6), use_cudnn)
        del e5, e6

        d9 = F.concat([self.dc9(e7), syn2])
        del e7, syn2

        d8 = F.relu(self.dc8(d9), use_cudnn)
        d7 = F.relu(self.dc7(d8), use_cudnn)
        del d9, d8

        d6 = F.concat([self.dc6(d7), syn1])
        del d7, syn1

        d5 = F.relu(self.dc5(d6), use_cudnn)
        d4 = F.relu(self.dc4(d5), use_cudnn)
        del d6, d5

        d3 = F.concat([self.dc3(d4), syn0])
        del d4, syn0

        d2 = F.relu(self.dc2(d3), use_cudnn)
        d1 = F.relu(self.dc1(d2), use_cudnn)
        del d3, d2

        d0 = self.dc0(d1)
        return d0
