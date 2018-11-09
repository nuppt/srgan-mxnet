from mxnet import gluon
from mxnet.gluon import nn

class ResnetBlock(gluon.nn.HybridBlock):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.HybridSequential()
        with self.name_scope():
            self.conv_block.add(
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm()
            )


    def hybrid_forward(self, F, x,*args, **kwargs):
        out = self.conv_block(x)
        return out + x

class SubpixelBlock(gluon.nn.HybridBlock):
    def __init__(self):
        super(SubpixelBlock, self).__init__()
        self.conv = nn.Conv2D(256, kernel_size=3, strides=1,padding=1)
        self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.conv(x)
        x = x.transpose([0, 2, 3, 1])
        batchsize,height,width,depth = x.shape
        x = x.reshape((batchsize, height , width, 2, 2, int(depth / 4)))
        x = x.transpose([0, 1,3,2,4,5])
        x = x.reshape((batchsize, height * 2, width * 2, int(depth / 4)))
        x = x.transpose([0, 3, 1, 2])
        x = self.relu(x)
        return x

class SRGenerator(gluon.nn.HybridBlock):
    def __init__(self):
        super(SRGenerator, self).__init__()
        self.conv1 = nn.Conv2D(64, in_channels= 3, kernel_size=3, strides=1,padding=1, activation='relu')
        self.res_block = nn.HybridSequential()
        with self.name_scope():
            for i in range(16):
                self.res_block.add(
                    ResnetBlock()
                )

            self.res_block.add(
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm()
            )
        self.subpix_block1 = SubpixelBlock()
        self.subpix_block2 = SubpixelBlock()
        self.conv4 = nn.Conv2D(3,kernel_size=1,strides=1,activation='tanh')

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.conv1(x)
        out = self.res_block(x)
        x = out + x
        x = self.subpix_block1(x)
        x = self.subpix_block2(x)
        x = self.conv4(x)
        return x

class ConvBlock(gluon.nn.HybridSequential):
    def __init__(self,filter_num,kernel_size=4,stride=2,padding=1):
        super(ConvBlock,self).__init__()
        self.model = nn.HybridSequential()
        with self.name_scope():
            self.model.add(
                nn.Conv2D(filter_num, kernel_size, stride,padding,use_bias=False),
                nn.BatchNorm(),
                nn.LeakyReLU(0.2),
            )

    def hybrid_forward(self, F, x,*args, **kwargs):
        return self.model(x)


class SRDiscriminator(gluon.nn.HybridBlock):
    def __init__(self):
        super(SRDiscriminator,self).__init__()
        self.model = nn.HybridSequential()
        self.res_block = nn.HybridSequential()
        df_dim = 64
        with self.name_scope():
            self.model.add(
                nn.Conv2D(df_dim, 4, 2, 1, in_channels=3),
                nn.LeakyReLU(0.2)
            )
            for i in [2,4,8,16,32]:
                self.model.add(ConvBlock(df_dim * i ))
            self.model.add(ConvBlock(df_dim * 16,1,1,padding=0))
            self.model.add(
                nn.Conv2D(df_dim * 8, 1, 1,use_bias=False),
                nn.BatchNorm()
            )
            self.res_block.add(
                ConvBlock(df_dim * 2, 1,1),
                ConvBlock(df_dim * 2, 3, 1),
                nn.Conv2D(df_dim * 8, 3, 1,use_bias=False),
                nn.BatchNorm()
            )
        self.lrelu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.dense = nn.Dense(1)

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.model(x)
        #23
        out = self.res_block(x)
        x = out + x
        x = self.lrelu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x