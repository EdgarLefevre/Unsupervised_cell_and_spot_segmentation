#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import tensorflow.keras as keras
import tensorflow.keras.layers as layers


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
        x = layers.concatenate([x, conc[i]])
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
