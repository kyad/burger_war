# -*- coding: utf-8 -*-
'''
ref: https://www.pynote.info/entry/keras-resnet-implementation
'''

from functools import reduce

from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2


def compose(*funcs):  # compose(f1, f2, f3) = f3(f2(f1))
    '''複数の層を結合する'''
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def baseConv2D(*args, **kwargs):
    '''ベースとなるconv2dを作成する'''
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(1.0e-4)
    }
    conv_kwargs.update(kwargs)
    return Conv2D(*args, **conv_kwargs)

def bn_relu_conv(*args, **kwargs):
    '''batch mormalization -> ReLU -> conv2dを作成する'''
    return compose(BatchNormalization(), Activation('relu'), baseConv2D(*args, **kwargs))

def shortcut(x, residual):
    '''shortcut connectionを作成する'''
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)

    if x_shape == residual_shape:
        # x と residual の形状が同じ場合、なにもしない。
        shortcut = x
    else:
        # x と residual の形状が異なる場合、線形変換を行い、形状を一致させる。
        stride_w = int(round(x_shape[1] / residual_shape[1]))
        stride_h = int(round(x_shape[2] / residual_shape[2]))

        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_w, stride_h),
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1.e-4))(x)
    return Add()([shortcut, residual])

def basic_block(filters, is_first_block_of_first_layer):
    '''bulding blockを作成する
        Args:
            filters: フィルター数
    '''
    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = baseConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(x)

        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        return shortcut(x, conv2)

    return f

def residual_blocks(block_function, filters, repetitions, is_first_layer):
    '''residual block を反復する構造を作成する。
        Args:
            block_function: residual block を作成する関数
            filters: フィルター数
            repetitions: residual block を何個繰り返すか。
            is_first_layer: max pooling 直後かどうか
    '''
    def f(x):
        for i in range(repetitions):
            x = block_function(filters=filters, is_first_block_of_first_layer=(i == 0 and is_first_layer))(x)
        return x

    return f

def resnet(input_shape=(16, 16, 7), num_layers=[3, 4, 3]):

    input = Input(shape=input_shape)
    conv1 = compose(
        baseConv2D(filters=32, kernel_size=(7, 7)),
        BatchNormalization(),
        Activation('relu'),
    )(input)

    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv1)

    block = pool1

    filters = [32, 64, 128]
    for i in range(3):
        block = residual_blocks(
            block_function=basic_block, filters=filters[i],
            repetitions=num_layers[i], is_first_layer=(i == 0)
        )(block)

    block = compose(
        BatchNormalization(),
        Activation('relu'),
        baseConv2D(filters=1, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation('tanh'),
    )(block)

    return Model(inputs=input, outputs=block)


# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# U-Net
#   https://qiita.com/koshian2/items/603106c228ac6b7d8356
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# U-Net
def create_block(input, chs):
    x = input
    for i in range(2):
        # オリジナルはpaddingなしだがサイズの調整が面倒なのでPaddingを入れる
        x = Conv2D(chs, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def create_unet(size=16, use_skip_connections=True, grayscale_inputs=True):

    #if grayscale_inputs: input = Input((96,96,1))
    #else:                input = Input((96,96,3))
    input = Input((16, 16, 7))

    # Encoder
    block1 = create_block(input, 64)
    x = MaxPooling2D(2)(block1)
    block2 = create_block(x, 128)
    x = MaxPooling2D(2)(block2)
    #block3 = create_block(x, 256)
    #x = MaxPooling2D(2)(block3)

    #x = create_block(x, 512)
    #x = Conv2DTranspose(256, kernel_size=2, strides=2)(x)
    #if use_skip_connections: x = Concatenate()([block3, x])
    x = create_block(x, 256)
    x = Conv2DTranspose(128, kernel_size=2, strides=2)(x)
    if use_skip_connections: x = Concatenate()([block2, x])
    x = create_block(x, 128)
    x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
    if use_skip_connections: x = Concatenate()([block1, x])
    x = create_block(x, 64)

    # output
    x = Conv2D(1, 1)(x)

    #x = Activation("linear")(x)
    x = Activation("tanh")(x)

    model  = Model(input, x)

    return model
