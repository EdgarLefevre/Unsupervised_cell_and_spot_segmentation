#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


# todo : implement l2 loss (reconstruction)

def edge_weights(flatten_image, rows, cols, std_intensity=3.0, std_position=1.0):
    """
    flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
    std_intensity : standard deviation for intensity
    std_position : standard devistion for position
    rows : rows of the original image (unflattened image)
    cols : cols of the original image (unflattened image)
    Output :
    weights :  2d tf array edge weights in the pixel graph
    Used parameters :
    n : number of pixels
    """
    A = outer_product(flatten_image, tf.ones_like(flatten_image))
    A_T = tf.transpose(A)
    intensity_weight = tf.exp(-1 * tf.square((tf.realdiv((A - A_T), std_intensity))))

    xx, yy = tf.meshgrid(tf.range(rows), tf.range(cols))
    xx = tf.reshape(xx, (rows * cols,))
    yy = tf.reshape(yy, (rows * cols,))
    A_x = outer_product(xx, tf.ones_like(xx))
    A_y = outer_product(yy, tf.ones_like(yy))

    xi_xj = A_x - tf.transpose(A_x)
    yi_yj = A_y - tf.transpose(A_y)

    sq_distance_matrix = tf.square(xi_xj) + tf.square(yi_yj)
    sq_distance_matrix = tf.cast(sq_distance_matrix, tf.float32)
    dist_weight = tf.exp(-1 * tf.realdiv(sq_distance_matrix, tf.square(std_position)))
    weight = tf.multiply(intensity_weight, dist_weight)
    return weight


def outer_product(v1, v2):
    """
    v1 : m*1 tf array
    v2 : m*1 tf array
    Output :
    v1 x v2 : m*m array
    """
    v1 = tf.reshape(v1, (-1,))
    v2 = tf.reshape(v2, (-1,))
    v1 = tf.expand_dims(v1, axis=0)
    v2 = tf.expand_dims(v2, axis=0)
    return tf.matmul(tf.transpose(v1), v2)


def numerator(k_class_prob, weights):
    """
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights n*n tensor
    """
    k_class_prob = tf.reshape(k_class_prob, (-1,))
    return tf.reduce_sum(tf.multiply(weights, outer_product(k_class_prob, k_class_prob)))


def denominator(k_class_prob, weights):
    """
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights	n*n tensor
    """
    k_class_prob = tf.cast(k_class_prob, tf.float32)
    k_class_prob = tf.reshape(k_class_prob, (-1,))
    return tf.reduce_sum(tf.multiply(weights, outer_product(k_class_prob, tf.ones(tf.shape(k_class_prob)))))


def soft_n_cut_loss(flatten_image, prob, k, rows, cols):
    """
    Inputs:
    prob : (rows*cols*k) tensor
    k : number of classes (integer)
    flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
    rows : number of the rows in the original image
    cols : number of the cols in the original image
    Output :
    soft_n_cut_loss tensor for a single image
    """

    loss = k
    weights = edge_weights(flatten_image, rows, cols)
    for t in range(k):
        loss = loss - (numerator(prob, weights) / denominator(prob, weights))
    return loss


def soft_n_cut_loss2(seg, weight, radius=5, K=2):
    cropped_seg = []
    sum_weight = tf.reduce_sum(weight)
    padded_seg = tf.pad(seg, [[radius - 1, radius - 1], [radius - 1, radius - 1], [radius - 1, radius - 1],
                              [radius - 1, radius - 1]])
    for m in tf.range((radius - 1) * 2 + 1, dtype=tf.int32):
        column = []
        for n in tf.range((radius - 1) * 2 + 1, dtype=tf.int32):
            column.append(tf.identity(padded_seg[:, :, m:m + seg.shape[2], n:n + seg.shape[3]]))
        cropped_seg.append(tf.stack(column, 4))
    cropped_seg = tf.stack(cropped_seg, 4)
    cropped_seg = tf.cast(cropped_seg, dtype=tf.float64)
    seg = tf.cast(seg, dtype=tf.float64)
    t_weight = tf.constant(weight)
    multi1 = tf.multiply(cropped_seg, t_weight)
    multi2 = tf.multiply(tf.reduce_sum(multi1), seg)
    multi3 = tf.multiply(sum_weight, seg)
    assocA = tf.reduce_sum(tf.reshape(multi2, (multi2.shape[1], multi2.shape[2], -1)))
    assocV = tf.reduce_sum(tf.reshape(multi3, (multi3.shape[1], multi3.shape[2], -1)))
    assoc = tf.reduce_sum(tf.realdiv(assocA, assocV))
    return tf.add(-assoc, K)  #  de base -assoc mais loss neg donc assoc
