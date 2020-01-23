import tensorflow as tf


def masked_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    # need this to fix wrong normalization of out-of-box SparseCategoricalCrossentropy
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                 reduction=tf.keras.losses.Reduction.SUM)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int32)
    mask_sum = tf.cast(tf.reduce_sum(mask), dtype=tf.float32)
    loss_sum = crossentropy(targets, logits, sample_weight=mask)
    return loss_sum / mask_sum
