#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import tensorflow.keras.applications as app
import tensorflow.keras.initializers as initializers
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def wnet(input_shape=(None, None, 3)):
    # Difference with original paper: padding 'valid vs same'
    conv_kernel_initializer = initializers.RandomNormal(stddev=0.01)

    input_flow = layers.Input(input_shape)
    # Encoder
    x = layers.Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(input_flow)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(
        128,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x_1 = layers.Conv2D(
        128,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x_1 = layers.Activation("relu")(x_1)

    x = layers.MaxPooling2D((2, 2))(x_1)
    x = layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x_2 = layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x_2 = layers.Activation("relu")(x_2)

    x = layers.MaxPooling2D((2, 2))(x_2)
    x = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x_3 = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x_3 = layers.Activation("relu")(x_3)

    x = layers.MaxPooling2D((2, 2))(x_3)
    x = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x_4 = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x_4 = layers.Activation("relu")(x_4)

    # Decoder 1
    x = layers.UpSampling2D((2, 2))(x_4)
    x = layers.concatenate([x_3, x])
    x = layers.Conv2D(
        256,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x_2, x])
    x = layers.Conv2D(
        128,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        128,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x_1, x])
    x = layers.Conv2D(
        64,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x)
    x = layers.Activation("relu")(x)

    # Decoder 2
    x_rb = layers.UpSampling2D((2, 2))(x_4)
    x_rb = layers.concatenate([x_3, x_rb])
    x_rb = layers.Conv2D(
        256,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x_rb)
    x_rb = layers.Activation("relu")(x_rb)
    x_rb = layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x_rb)
    x_rb = layers.Activation("relu")(x_rb)

    x_rb = layers.UpSampling2D((2, 2))(x_rb)
    x_rb = layers.concatenate([x_2, x_rb])
    x_rb = layers.Conv2D(
        128,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x_rb)
    x_rb = layers.Activation("relu")(x_rb)
    x_rb = layers.Conv2D(
        128,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x_rb)
    x_rb = layers.Activation("relu")(x_rb)

    x_rb = layers.UpSampling2D((2, 2))(x_rb)
    x_rb = layers.concatenate([x_1, x_rb])
    x_rb = layers.Conv2D(
        64,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x_rb)
    x_rb = layers.Activation("relu")(x_rb)
    x_rb = layers.Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x_rb)
    x_rb = layers.Activation("relu")(x_rb)
    x_rb = layers.Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
    )(x_rb)
    x_rb = layers.Activation("relu")(x_rb)
    x_rb = layers.Conv2D(
        1,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
        activation="sigmoid",
    )(
        x_rb
    )  # Sigmoid activation

    # Multiplication
    x = layers.multiply([x, x_rb])
    x = layers.Conv2D(
        1,
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer=conv_kernel_initializer,
        activation="relu",
    )(x)

    model = models.Model(inputs=input_flow, outputs=x)

    front_end = app.VGG16(
        weights="imagenet", include_top=False
    )  # apply vgg weights to model, maybe we don't need this
    weights_front_end = []
    for layer in front_end.layers:
        if "conv" in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(model.layers)):
        if counter_conv >= 13:
            break
        if "conv" in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1

    return model
