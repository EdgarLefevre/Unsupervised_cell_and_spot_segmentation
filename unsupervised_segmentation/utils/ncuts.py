import os

import numpy as np
import tensorflow as tf

os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


"""
ce code vient de : https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/
"""


def sparse_tensor_dense_tensordot(sp_a, b, axes):
    def _tensordot_reshape(a, axes, flipped=False):
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(axes < 0, tf.int32) * (
                axes + rank_a
            )
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            # perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    # def _tensordot_axes(a, axes):
    #     a_shape = a.get_shape()
    #     if isinstance(axes, tf.compat.integral_types):
    #         if axes < 0:
    #             raise ValueError("'axes' must be at least 0.")
    #         if a_shape.ndims is not None:
    #             if axes > a_shape.ndims:
    #                 raise ValueError("'axes' must not be larger than the number of "
    #                                  "dimensions of tensor %s." % a)
    #             return (list(range(a_shape.ndims - axes, a_shape.ndims)),
    #                     list(range(axes)))
    #         else:
    #             rank = tf.rank(a)
    #             return (range(rank - axes, rank, dtype=tf.int32),
    #                     range(axes, dtype=tf.int32))
    #     elif isinstance(axes, (list, tuple)):
    #         if len(axes) != 2:
    #             raise ValueError("'axes' must be an integer or have length 2.")
    #         a_axes = axes[0]
    #         b_axes = axes[1]
    #         if isinstance(a_axes, tf.compat.integral_types) and \
    #                 isinstance(b_axes, tf.compat.integral_types):
    #             a_axes = [a_axes]
    #             b_axes = [b_axes]
    #         if len(a_axes) != len(b_axes):
    #             raise ValueError(
    #                 "Different number of contraction axes 'a' and 'b', %s != %s." %
    #                 (len(a_axes), len(b_axes)))
    #         return a_axes, b_axes
    #     else:
    #         axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
    #     return axes[0], axes[1]

    def _sparse_tensordot_reshape(a, axes, flipped=False):
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.sparse.reshape(tf.sparse.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(axes < 0, tf.int32) * (
                axes + rank_a
            )
            # print(sess.run(rank_a), sess.run(axes))
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            # perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.sparse_reshape(tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _sparse_tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError(
                        "'axes' must not be larger than the number of "
                        "dimensions of tensor %s." % a
                    )
                return (
                    list(range(a_shape.ndims - axes, a_shape.ndims)),
                    list(range(axes)),
                )
            else:
                rank = tf.rank(a)
                return (
                    range(rank - axes, rank, dtype=tf.int32),
                    range(axes, dtype=tf.int32),
                )
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and isinstance(
                b_axes, tf.compat.integral_types
            ):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                    "Different number of contraction axes 'a' and 'b', %s != %s."
                    % (len(a_axes), len(b_axes))
                )
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    # start exec here
    b = tf.convert_to_tensor(b, name="b")
    sp_a_axes, b_axes = _sparse_tensordot_axes(sp_a, axes)
    sp_a_reshape, sp_a_free_dims, sp_a_free_dims_static = _sparse_tensordot_reshape(
        sp_a, sp_a_axes
    )
    b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(b, b_axes, True)
    # print(100*'-')
    # print(sp_a)
    # print(b)
    # print(100 * '-')
    ab_matmul = tf.sparse.sparse_dense_matmul(sp_a_reshape, b_reshape)
    if isinstance(sp_a_free_dims, list) and isinstance(b_free_dims, list):
        return tf.reshape(ab_matmul, sp_a_free_dims + b_free_dims)
    else:
        sp_a_free_dims = tf.convert_to_tensor(sp_a_free_dims, dtype=tf.int32)
        b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.int32)
        product = tf.reshape(ab_matmul, tf.concat([sp_a_free_dims, b_free_dims], 0))
        if sp_a_free_dims_static is not None and b_free_dims_static is not None:
            product.set_shape(sp_a_free_dims_static + b_free_dims_static)
        return product


def circular_neighbor(index_centor, r, image_shape):
    xc, yc = index_centor
    x = np.arange(0, 2 * r + 1)
    y = np.arange(0, 2 * r + 1)
    in_circle = ((x[np.newaxis, :] - r) ** 2 + (y[:, np.newaxis] - r) ** 2) < r ** 2
    in_cir_x, in_cir_y = np.nonzero(in_circle)
    in_cir_x += xc - r
    in_cir_y += yc - r
    x_in_array = (0 <= in_cir_x) * (in_cir_x < image_shape[0])
    y_in_array = (0 <= in_cir_y) * (in_cir_y < image_shape[1])
    in_array = x_in_array * y_in_array
    return in_cir_x[in_array], in_cir_y[in_array]


def gaussian_neighbor(image_shape, sigma_X=4, r=5):
    row_lst, col_lst, val_lst = [], [], []
    for i, (a, b) in enumerate(np.ndindex(*image_shape)):
        neighbor_x, neighbor_y = circular_neighbor((a, b), r, image_shape)
        neighbor_value = np.exp(
            -((neighbor_x - a) ** 2 + (neighbor_y - b) ** 2) / sigma_X ** 2
        )
        ravel_index = np.ravel_multi_index([neighbor_x, neighbor_y], image_shape)
        row_lst.append(np.array([i] * len(neighbor_x)))
        col_lst.append(ravel_index)
        val_lst.append(neighbor_value)
    rows = np.hstack(row_lst)
    cols = np.hstack(col_lst)
    indeces = np.vstack([rows, cols]).T.astype(np.int64)
    vals = np.hstack(val_lst).astype(np.float)
    return indeces, vals


def brightness_weight(image, neighbor_filter, weight_shapes, sigma_I=0.05):
    indeces, vals = neighbor_filter
    rows = indeces[:, 0]
    cols = indeces[:, 1]
    image = tf.reshape(image, shape=(-1, weight_shapes))
    image = tf.transpose(image, [1, 0])
    i_embedding = tf.nn.embedding_lookup(image, rows)
    j_embedding = tf.nn.embedding_lookup(image, cols)
    Fi = tf.transpose(i_embedding, [1, 0])  # [B, #elements]
    Fj = tf.transpose(j_embedding, [1, 0])  # [B, #elements]
    bright_weight = tf.exp(-((Fi - Fj) ** 2) / sigma_I ** 2) * vals
    bright_weight = tf.transpose(bright_weight, [1, 0])  # [#elements, B]
    return bright_weight


def convert_to_batchTensor(indeces, batch_values, dense_shape):
    batch_size = tf.cast(tf.shape(batch_values)[1], tf.int64)
    num_element = tf.cast(tf.shape(indeces)[0], tf.int64)
    # Expand indeces, values
    tile_indeces = tf.tile(indeces, tf.stack([batch_size, 1]))
    tile_batch = tf.range(batch_size, dtype=tf.int64)
    tile_batch = tf.tile(tf.expand_dims(tile_batch, axis=1), tf.stack([1, num_element]))
    tile_batch = tf.reshape(tile_batch, [-1, 1])

    # Expand dense_shape
    new_indeces = tf.concat([tile_batch, tile_indeces], axis=1)
    new_batch_values = tf.reshape(batch_values, [-1])
    new_dense_shape = tf.concat(
        [
            tf.cast(tf.reshape(batch_size, [-1]), tf.int32),
            tf.cast(dense_shape, tf.int32),
        ],
        axis=0,
    )
    new_dense_shape = tf.cast(new_dense_shape, tf.int64)
    # Construct 3D tensor [B, W*H, W*H]
    batchTensor = tf.SparseTensor(new_indeces, new_batch_values, new_dense_shape)
    return batchTensor


def sycronize_axes(tensor, axes, tensor_dims=None):
    # Swap axes to head dims
    if tensor_dims is None:
        tensor_dims = len(tensor.get_shape().as_list())
    perm_axes = list(axes)
    perm_axes.extend([i for i in range(tensor_dims) if i not in axes])
    perm_tensor = tf.transpose(tensor, perm_axes)

    # Expand
    contract_axis_0_len = tf.shape(perm_tensor)[0]
    contract_axis_len = len(axes)
    diag_slice = tf.range(contract_axis_0_len)
    diag_slice = tf.expand_dims(diag_slice, axis=1)
    diag_slice = tf.tile(diag_slice, tf.stack([1, contract_axis_len]))

    # Slice diagonal elements
    syn_tensor = tf.gather_nd(perm_tensor, diag_slice)
    return syn_tensor


def soft_ncut(image, image_segment, image_weights):
    batch_size = tf.shape(image)[0]
    num_class = tf.shape(image_segment)[-1]
    image_shape = image.shape
    weight_size = image_shape[1] * image_shape[2]
    image_segment = tf.transpose(image_segment, [0, 3, 1, 2])  # [B, K, H, W]
    image_segment = tf.reshape(
        image_segment, tf.stack([batch_size, num_class, weight_size])
    )  # [B, K, H*W]

    # Dis-association
    # [B0, H*W, H*W] @ [B1, K1, H*W] contract on [[2],[2]] = [B0, H*W, B1, K1]
    if len(image_weights.shape) > 3:
        image_weights = tf.sparse.reshape(
            image_weights,
            [image_weights.shape[0], image_weights.shape[2], image_weights.shape[3]],
        )
    W_Ak = sparse_tensor_dense_tensordot(image_weights, image_segment, axes=[[2], [2]])
    W_Ak = tf.transpose(W_Ak, [0, 2, 3, 1])  # [B0, B1, K1, H*W]
    W_Ak = sycronize_axes(W_Ak, [0, 1], tensor_dims=4)  # [B0=B1, K1, H*W]
    # [B1, K1, H*W] @ [B2, K2, H*W] contract on [[2],[2]] = [B1, K1, B2, K2]
    dis_assoc = tf.tensordot(W_Ak, image_segment, axes=[[2], [2]])
    dis_assoc = sycronize_axes(dis_assoc, [0, 2], tensor_dims=4)  # [B1=B2, K1, K2]
    dis_assoc = sycronize_axes(dis_assoc, [1, 2], tensor_dims=3)  # [K1=K2, B1=B2]
    dis_assoc = tf.transpose(dis_assoc, [1, 0])  # [B1=B2, K1=K2]
    dis_assoc = tf.identity(dis_assoc, name="dis_assoc")

    # Association
    # image_segment: [B0, K0, H*W]
    sum_W = tf.sparse.reduce_sum(image_weights, axis=2)  # [B1, W*H]

    assoc = tf.tensordot(
        tf.cast(image_segment, dtype=tf.float32), sum_W, axes=[2, 1]
    )  # [B0, K0, B1]
    assoc = sycronize_axes(assoc, [0, 2], tensor_dims=3)  # [B0=B1, K0]
    assoc = tf.identity(assoc, name="assoc")

    # Soft NCut
    eps = 1e-6
    soft_ncut = tf.cast(num_class, tf.float32) - tf.reduce_sum(
        tf.cast((dis_assoc + eps), dtype=tf.float32) / (assoc + eps), axis=1
    )
    return soft_ncut


def process_weight(img):
    img_size = img.shape[0]
    indeces, vals = gaussian_neighbor((img_size, img_size))
    weight_shapes = np.prod((img_size, img_size)).astype(np.int64)
    weight_size = tf.constant([weight_shapes, weight_shapes])
    b = brightness_weight(img, (indeces, vals), weight_shapes)
    return convert_to_batchTensor(indeces, b, weight_size)


def process_weight_multi(images):
    res = []
    for img in images:
        img_size = img.shape[0]
        indeces, vals = gaussian_neighbor((img_size, img_size))
        weight_shapes = np.prod((img_size, img_size)).astype(np.int64)
        weight_size = tf.constant([weight_shapes, weight_shapes])
        b = brightness_weight(img, (indeces, vals), weight_shapes)
        res.append(convert_to_batchTensor(indeces, b, weight_size))
    return tf.sparse.concat(axis=0, sp_inputs=res)


if __name__ == "__main__":
    img_size = 128
    image = tf.zeros((img_size, img_size))
    indeces, vals = gaussian_neighbor((img_size, img_size))

    weight_shapes = np.prod((img_size, img_size)).astype(np.int64)
    weight_size = tf.constant([weight_shapes, weight_shapes])
    b = brightness_weight(image, (indeces, vals), weight_shapes)

    print(100 * "-")
    print("bright weigths")
    print(b)
    print(b.shape)

    new_b = convert_to_batchTensor(indeces, b, weight_size)

    print(100 * "-")
    print("new")
    print(new_b)
    print(new_b.shape)
    # print("\n" * 5)
    #
    # res = soft_ncut(np.ones((1, img_size, img_size, 1)),
    #                 np.ones((1, img_size, img_size, 2)),
    #                 new_b)
    # print(res)
    # img_size = 128
    # images = tf.zeros((5, img_size, img_size))
    # res = process_weight(images)
    # print(res)
