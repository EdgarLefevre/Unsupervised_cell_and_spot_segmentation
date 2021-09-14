#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers


def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input: down-dim feature map
    :param out_size: output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding="same")(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def expend_as(tensor, rep):
    return layers.Lambda(
        lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
        arguments={"repnum": rep},
    )(tensor)


def attention_block(x, gating, inter_shape):
    """
    From https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831 ;
    did some adaptation, not sure for now if it's working
    """
    shape_x = x.shape  # 16,16,64
    shape_g = gating.shape  # 8,8,32

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(1, 1), padding="same")(
        x
    )  # 16,16,32
    shape_theta_x = theta_x.shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding="same")(gating)  # 8,8,32
    upsample_g = layers.Conv2DTranspose(
        inter_shape,
        (3, 3),
        strides=(1, 1),
        # strides=(2, 2),
        padding="same",
    )(
        phi_g
    )  # 16,16,32

    concat_xg = layers.add([upsample_g, theta_x])  # 16,16,64
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding="same")(act_xg)  # 16,16,1
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    shape_sigmoid = sigmoid_xg.shape  # 16,16,1
    # upsample_psi = layers.UpSampling2D(size=(2, 2))(sigmoid_xg)  # 32,32,1
    upsample_psi = expend_as(sigmoid_xg, shape_x[3])  # 16,16,64

    y = layers.multiply([upsample_psi, x])  # 16,16,64

    result = layers.Conv2DTranspose(shape_x[3], (2, 2), strides=(2, 2), padding="same")(
        y
    )
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def attention_block2(x, gating, inter_shape):
    """
    From https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831 ;
    did some adaptation, not sure for now if it's working
    """
    theta_x = layers.Conv2D(inter_shape, kernel_size=1, strides=2, padding="same")(
        x
    )  # 8,8,32
    phi_g = layers.Conv2D(inter_shape, kernel_size=1, padding="same")(
        gating
    )  # 16,16,32
    concat_xg = layers.add([phi_g, theta_x])  # 8,8,32
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding="same")(act_xg)  # 8,8,1
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    upsample_psi = layers.UpSampling2D(size=(2, 2))(sigmoid_xg)  # 16,16,1
    y = layers.multiply([upsample_psi, x])  # 16,16,32
    return y


def block_down(
    inputs, filters, drop=0.3, w_decay=0.0001, kernel_size=3, separable=False
):
    if separable:
        x = layers.SeparableConv2D(
            filters,
            (kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
            activation="elu",
        )(inputs)
        c = layers.SeparableConv2D(
            filters,
            (kernel_size, kernel_size),
            activation="elu",
            kernel_initializer="he_normal",
            padding="same",
        )(x)
    else:
        x = layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
            activation="elu",
        )(inputs)
        c = layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            activation="elu",
            kernel_initializer="he_normal",
            padding="same",
        )(x)
    p = layers.BatchNormalization()(c)
    p = layers.Dropout(drop)(p)
    p = layers.MaxPooling2D((2, 2))(p)
    return p, c


def bridge(inputs, filters, drop=0.3, kernel_size=3):
    x = layers.SeparableConv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="elu",
    )(inputs)
    x = layers.SeparableConv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="elu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(drop)(x)
    return x


def block_up(
    input, conc, filters, drop=0.3, w_decay=0.0001, kernel_size=3, separable=False
):
    x = layers.Conv2DTranspose(
        filters,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(input)
    for i in range(len(conc)):
        gat = gating_signal(input, filters)
        print(conc[i].shape)
        print(gat.shape)
        print(100 * "-")
        # att = attention_block(input, gat, filters)
        att = attention_block2(conc[i], gat, filters)
<<<<<<< HEAD
        x = layers.concatenate([x, att])
=======
        x = layers.concatenate([x, conc[i], att])
>>>>>>> bbbf7fe9f0a0d93f46380591fcbf2ed6341df405
    if separable:
        x = layers.SeparableConv2D(
            filters,
            (kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
            activation="elu",
        )(x)
        x = layers.SeparableConv2D(
            filters,
            (kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
            activation="elu",
        )(x)
    else:
        x = layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
            activation="elu",
        )(x)
        x = layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
            activation="elu",
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(drop)(x)
    return x


def unet(input_size, enc, name):
    input = layers.Input(input_size)
    filters = 8
    # down
    d1, c1 = block_down(input, filters=filters)
    d2, c2 = block_down(d1, filters=filters * 2, separable=True)
    d3, c3 = block_down(d2, filters=filters * 4, separable=True)
    # d4, c4 = block_down(d3, filters=filters * 8, separable=True)

    # bridge
    b = bridge(d3, filters=filters * 8)

    # up
    # u4 = block_up(input=b, filters=filters * 8, conc=[c4], separable=True)
    u3 = block_up(input=b, filters=filters * 4, conc=[c3], separable=True)
    u2 = block_up(input=u3, filters=filters * 2, conc=[c2], separable=True)
    u1 = block_up(input=u2, filters=filters, conc=[c1])
    if enc:
        output = layers.Conv2D(2, (1, 1))(u1)
    else:
        output = layers.Conv2D(1, (1, 1))(u1)
    return keras.Model(input, output, name=name)


def u_enc(input_size):
    return unet(input_size, True, name="encoder")


def u_dec(input_size):
    return unet(input_size, False, name="decoder")


def wnet(input_shape):
    input = layers.Input(input_shape)
    uenc = u_enc(input_shape)
    x = uenc(input)
    udec = u_dec((x.shape[1], x.shape[2], x.shape[3]))
    output = udec(x)
    model = keras.Model(inputs=input, outputs=output, name="wnet")
    return uenc, model


# todo: rename things
