import tensorflow as tf


class AMSGrad(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="AMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            m = tf.compat.v1.get_variable(f"{var.op.name}/m", shape=var.shape, initializer=tf.zeros_initializer(), trainable=False)
            v = tf.compat.v1.get_variable(f"{var.op.name}/v", shape=var.shape, initializer=tf.zeros_initializer(), trainable=False)
            v_hat = tf.compat.v1.get_variable(f"{var.op.name}/v_hat", shape=var.shape, initializer=tf.zeros_initializer(), trainable=False)

            m_t = self._beta1 * m + (1 - self._beta1) * grad
            v_t = self._beta2 * v + (1 - self._beta2) * tf.square(grad)
            v_hat_t = tf.maximum(v_hat, v_t)

            var_update = tf.assign_sub(var, self._lr * m_t / (tf.sqrt(v_hat_t) + self._epsilon))
            assignments.extend([tf.assign(m, m_t), tf.assign(v, v_t), tf.assign(v_hat, v_hat_t), var_update])
        return tf.group(*assignments, name=name)
