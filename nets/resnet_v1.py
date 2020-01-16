import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock, HybridSequential, BatchNorm
from gluoncv.model_zoo import get_model
from mxnet.gluon.contrib.nn import SyncBatchNorm
from gluoncv.model_zoo.xception import SeparableConv2d


# helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


def _conv1x1(channels):
    return nn.Conv2D(channels, kernel_size=1, strides=1)

'''
Enhanced_Attention_Module
'''


class Downblock_V2(HybridBlock):
    def __init__(self, norm_layer, channels, norm_kwargs=None):
        super(Downblock_V2, self).__init__()
        with self.name_scope():
            # 使用分组卷积
            self.dwconv_3 = nn.Conv2D(channels=channels, groups=channels, kernel_size=3, strides=2,
                                      dilation=2, padding=2,
                                      use_bias=False)
            self.dwconv_7 = nn.Conv2D(channels=channels, groups=channels, kernel_size=3, dilation=4, strides=2,
                                      padding=4, use_bias=False)
            self.bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs))

    def hybrid_forward(self, F, x):
        '''
        :type F:mx.symbol
        '''
        x_1 = self.dwconv_3(x)
        x_2 = self.dwconv_7(x)
        x = F.concat(x_1, x_2)
        x = self.bn(x)
        return x


class Downblock_V3(HybridBlock):
    def __init__(self, norm_layer, channels, norm_kwargs=None):
        super(Downblock_V3, self).__init__()
        with self.name_scope():
            # 使用分组卷积
            self.dwconv_3 = SeparableConv2d(inplanes=channels, planes=channels, kernel_size=3, dilation=2, stride=2,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dwconv_7 = SeparableConv2d(inplanes=channels, planes=channels, kernel_size=3, dilation=4, stride=2,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs
                                            )
    def hybrid_forward(self, F, x):
        '''
        :type F:mx.symbol
        '''
        x_1 = self.dwconv_3(x)
        x_2 = self.dwconv_7(x)
        x = F.concat(x_1, x_2)
        return x


class Enhanced_Spatial_Attention(HybridBlock):
    def __init__(self, norm_layer, channels, kernel_size, dilation=1, norm_kwargs=None):
        super(Enhanced_Spatial_Attention, self).__init__()
        self.kernel_size = kernel_size
        with self.name_scope():
            # downop的操作
            self.downop = Downblock_V3(norm_layer, channels=channels, norm_kwargs=norm_kwargs)
            # 添加空洞卷积 可以查看是否需要修改成分组卷积
            self.conv = SeparableConv2d(inplanes=channels*2, planes=channels,
                                        kernel_size=3, dilation=dilation,
                                        norm_layer=norm_layer,norm_kwargs=norm_kwargs)
            # self.conv = nn.Conv2D(channels, groups=channels, kernel_size=3, padding=dilation, dilation=dilation,
            #                       use_bias=False)
            # self.conv_1x1 = _conv1x1(channels=channels)

    def hybrid_forward(self, F, x):
        '''
        :type F:mx.symbol
        '''
        x = self.downop(x)
        x = self.conv(x)
        x = F.contrib.BilinearResize2D(data=x, height=self.kernel_size, width=self.kernel_size)
        # x=F.sigmoid(x)
        return x


class Enhanced_Channel_Attenion(HybridBlock):
    def __init__(self, norm_layer, channels, reduction_ratio=16, norm_kwargs=None):
        super(Enhanced_Channel_Attenion, self).__init__()
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.max_pool = nn.GlobalMaxPool2D()
            self.down_op = nn.Conv2D(1, kernel_size=(2, 1))
            self.gate_c = nn.HybridSequential()
            self.gate_c.add(nn.Dense(channels // reduction_ratio, use_bias=False))
            self.gate_c.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.gate_c.add(nn.Activation('relu'))
            self.gate_c.add(nn.Dense(channels, use_bias=False))

    def hybrid_forward(self, F, x):
        '''
        :type F:mx.symbol
        '''
        x_avg = F.flatten(self.avg_pool(x)).expand_dims(axis=1).expand_dims(axis=1)
        x_max = F.flatten(self.max_pool(x)).expand_dims(axis=1).expand_dims(axis=1)
        x = F.concat(x_avg, x_max, dim=2)
        x = self.down_op(x)
        x = F.flatten(x)
        x = self.gate_c(x)
        # x=F.sigmoid(x)
        # x=F.elemwise_add(x,F.ones_like(x))
        return x


class Enhanced_Attention_Module(HybridBlock):
    def __init__(self, norm_layer, channels, kernel_size, norm_kwargs=None):
        super(Enhanced_Attention_Module, self).__init__()
        with self.name_scope():
            self.channel_att = Enhanced_Channel_Attenion(norm_layer, channels, reduction_ratio=16,
                                                         norm_kwargs=norm_kwargs)
            self.spatial_att = Enhanced_Spatial_Attention(norm_layer, channels, kernel_size=kernel_size,
                                                          norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x):
        '''
        :type F:mx.symbol
        '''
        att_c = self.channel_att(x).expand_dims(axis=2).expand_dims(axis=2)
        att_s = self.spatial_att(x)
        w = F.sigmoid(F.broadcast_mul(att_c, att_s))
        # x = F.broadcast_mul(x, self.channel_att(x).expand_dims(axis=2).expand_dims(axis=2))
        # x = F.broadcast_mul(x, self.spatial_att(x))

        return w


'''
Bottleneck实现
'''


class BottleneckV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, use_ones=False, use_ge=False, ge_size=None, use_eam=False,
                 use_cbam=False,
                 norm_layer=BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.use_ones = use_ones
        # 设置主体部分
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels // 4, kernel_size=1, strides=stride))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels // 4, 1, channels // 4))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        # Enhanced_Attention_Module
        if use_eam and ge_size:
            self.eam = Enhanced_Attention_Module(norm_layer, channels, kernel_size=ge_size, norm_kwargs=norm_kwargs)
        else:
            self.eam = None
        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        '''
        :type F:mx.symbol
        '''
        residual = x
        x = self.body(x)
        if self.eam:
            w = self.eam(x)
            x = F.elemwise_mul(x, w)
            # x=self.eam(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = F.Activation(x + residual, act_type='relu')
        return x


# Nets
class ResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, classes: int = None, thumbnail=False,
                 last_gamma=False, use_se=False, use_ones=False, use_ge=False, use_eam=False, use_cbam=False,
                 norm_layer=BatchNorm,
                 norm_kwargs=None, use_dropout=False,
                 **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert isinstance(classes, int)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                if use_ge or use_eam:
                    ge_size = [64, 32, 16, 8]
                    self.features.add(self._make_layer(block, num_layer, channels[i + 1],
                                                       stride, i + 1, in_channels=channels[i],
                                                       last_gamma=last_gamma, use_se=use_se, use_ones=use_ones,
                                                       use_ge=use_ge, ge_size=ge_size[i], use_eam=use_eam,
                                                       use_cbam=use_cbam,
                                                       norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                else:
                    self.features.add(self._make_layer(block, num_layer, channels[i + 1],
                                                       stride, i + 1, in_channels=channels[i],
                                                       last_gamma=last_gamma, use_se=use_se, use_ones=use_ones,
                                                       use_ge=use_ge, ge_size=None, use_eam=use_eam,
                                                       use_cbam=use_cbam,
                                                       norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.GlobalAvgPool2D())
            if use_dropout:
                self.dropout = nn.Dropout(rate=0.5)
            else:
                self.dropout = None
            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, use_ones=False, use_ge=False, ge_size=None, use_eam=False,
                    use_cbam=False,
                    norm_layer=BatchNorm,
                    norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, use_ones=use_ones, use_ge=use_ge, ge_size=ge_size,
                            use_eam=use_eam, use_cbam=use_cbam,
                            prefix='',
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers - 1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, use_ones=use_ones, use_ge=use_ge, ge_size=ge_size,
                                use_eam=use_eam, use_cbam=use_cbam,
                                prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.output(x)
        return x


def resnet50_v1(*args, **kwargs):
    model_name = 'resnet50_v1'
    net = get_model(model_name, pretrained=True)
    with net.name_scope():
        net.output = nn.Dense(kwargs['num_classes'])
    return net


def resnet101_v1(*args, **kwargs):
    model_name = 'resnet101_v1'
    net = get_model(model_name, pretrained=True)
    with net.name_scope():
        net.output = nn.Dense(kwargs['num_classes'])
    return net


def resnet50_v1_eam(*args, **kwargs):
    net = ResNetV1(block=BottleneckV1, use_eam=True, layers=[3, 4, 6, 3],
                   norm_layer=nn.BatchNorm,
                   use_dropout=False,
                   channels=[64, 256, 512, 1024, 2048], classes=kwargs['num_classes'])
    return net


def resnet101_v1_eam(*args, **kwargs):
    net = ResNetV1(block=BottleneckV1, use_eam=True, layers=[3, 4, 23, 3],
                   use_dropout=False,
                   channels=[64, 256, 512, 1024, 2048], classes=kwargs['num_classes'])
    return net


if __name__ == '__main__':
    net = ResNetV1(block=BottleneckV1, use_eam=True, layers=[3, 4, 6, 3],
                   channels=[64, 256, 512, 1024, 2048], use_dropout=False,
                   norm_layer=BatchNorm,
                   classes=30)
    print(net)
    net.initialize()
    x = nd.random.randn(1, 3, 256, 256)
    print(net(x))
