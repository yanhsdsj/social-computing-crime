import tensorflow as tf
import numpy as np

def cross_entropy(null_val=0.0):
    import tensorflow as tf

    def loss(preds, labels):
        preds = tf.clip_by_value(preds, 1e-6, 1.0 - 1e-6)  # 避免 log(0)
        xent = - labels * tf.log(preds) - (1 - labels) * tf.log(1 - preds)
        return tf.reduce_mean(xent)

    return loss

def masked_mae_loss(scaler, null_val=np.nan):
    def loss(preds, labels):
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if np.isnan(null_val):
            mask = ~tf.math.is_nan(labels)
        else:
            mask = tf.not_equal(labels, null_val)
        mask = tf.cast(mask, tf.float32)
        mask /= tf.reduce_mean(mask)
        loss = tf.abs(tf.subtract(preds, labels))
        loss = loss * mask
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        return tf.reduce_mean(loss)
    return loss

# def cross_entropy_with_class_weights(class_weights, null_val=0.0):
#     """
#     适用于类别不平衡的加权交叉熵损失函数。
#     :param class_weights: array/list，长度为类别数，每个类别的权重
#     """
#     class_weights = tf.constant(class_weights, dtype=tf.float32)

#     def loss(preds, labels):
#         preds = tf.clip_by_value(preds, 1e-6, 1.0 - 1e-6)
#         # preds, labels shape: (batch_size, num_nodes, num_classes)
#         bce = - labels * tf.math.log(preds) - (1 - labels) * tf.math.log(1 - preds)

#         weights = tf.reshape(class_weights, (1, 1, -1))  # 广播匹配 shape
#         weighted_bce = bce * weights

#         return tf.reduce_mean(weighted_bce)

#     return loss