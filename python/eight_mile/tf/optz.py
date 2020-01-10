import logging
import tensorflow as tf
from eight_mile.utils import get_version, exporter
logger = logging.getLogger('mead.layers')


__all__ = []
export = exporter(__all__)


@export
class AdamWOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.

    Modified from: https://github.com/google-research/bert/blob/master/optimization.py
    This does the weight decay slightly differently from PyTorch version, putting it before the update
    """

    def __init__(self,
                 learning_rate,
                 weight_decay=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 name="AdamWOptimizer"):
        """Constructs a AdamWOptimizer."""
        super(AdamWOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def _get_variable_name(self, param_name):
        import re
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            update += self.weight_decay * param
            update_with_lr = self.learning_rate * update
            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)


@export
def optimizer(loss_fn, **kwargs):

    global_step = tf.train.get_or_create_global_step()
    clip = kwargs.get('clip', None)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('lr', kwargs.get('eta', 0.01))
    lr_scheduler = create_lr_scheduler(**kwargs)
    decay_fn = None
    colocate_gradients_with_ops = bool(kwargs.get('colocate_gradients_with_ops', False))
    sgd_mom = float(kwargs.get('mom', 0.9))
    if optim == 'adadelta':
        rho = float(kwargs.get('rho', 0.95))
        eps = float(kwargs.get('epsilon', 1e-6))
        logger.info('adadelta(eta=%f, rho=%f, epsilon=%f)', eta, rho, eps)
        optz = lambda lr: tf.train.AdadeltaOptimizer(lr, rho, eps)
    elif optim == 'adam':
        beta1 = float(kwargs.get('beta1', 0.9))
        beta2 = float(kwargs.get('beta2', 0.999))
        eps = float(kwargs.get('epsilon', 1e-8))
        logger.info('adam(eta=%f beta1=%f, beta2=%f, eps=%f)', eta, beta1, beta2, eps)
        optz = lambda lr: tf.train.AdamOptimizer(lr, beta1, beta2, eps)
    elif optim == 'adamw':
        wd = float(kwargs.get('weight_decay', 0))
        beta1 = float(kwargs.get('beta1', 0.9))
        beta2 = float(kwargs.get('beta2', 0.999))
        eps = float(kwargs.get('epsilon', 1e-8))
        logger.info('adamw(eta=%f beta1=%f, beta2=%f, eps=%f)', eta, beta1, beta2, eps)
        optz = lambda lr: AdamWOptimizer(lr, wd, beta1, beta2, eps)
    elif optim == 'rmsprop':
        # Get mom again with difference default
        mom = float(kwargs.get('mom', 0.0))
        logger.info('rmsprop(eta=%f, mom=%f)', eta, mom)
        optz = lambda lr: tf.train.RMSPropOptimizer(lr, momentum=mom)
    elif sgd_mom > 0:
        logger.info('sgd-mom(eta=%f, mom=%f)', eta, sgd_mom)
        optz = lambda lr: tf.train.MomentumOptimizer(lr, sgd_mom)
    else:
        logger.info('sgd(eta=%f)', eta)
        optz = lambda lr: tf.train.GradientDescentOptimizer(lr)

    logger.info('clip gradients at %s', clip)
    return global_step, tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz,
                                                        colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                        clip_gradients=clip, learning_rate_decay_fn=lr_scheduler,
                                                        increment_global_step=True)
# https://www.tensorflow.org/guide/eager
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
    y = tf.identity(x)

    def grad_fn(dresult):
        return [tf.clip_by_norm(dresult, norm), None]

    return y, grad_fn

# Warning, sparse update ops dont work on GPU
# In TF 2 this leads to errors, particularly with SGD w/ Momentum and Adadelta
# https://github.com/tensorflow/tensorflow/issues/31291
@export
class EagerOptimizer(object):

    def __init__(self, loss, optimizer=None, **kwargs):
        self.loss = loss
        self.global_step = tf.Variable(0)
        if 'lr_function' in kwargs:
            lr_function = kwargs['lr_function']
        else:
            if 'lr_scheduler_type' not in kwargs:
                kwargs['lr_scheduler_type'] = 'default'
            lr_function = create_lr_scheduler(**kwargs)
        #decay_fn = None
        # Right now this option is pointless since sparse updates dont work on GPU.  We just turn it off
        sgd_mom = float(kwargs.get('mom', 0.9))
        self.clip = kwargs.get('clip', 100)

        if optimizer:
            self.optimizer = optimizer
        else:
            optim = kwargs.get('optim', 'sgd')
            lr = kwargs.get('lr', kwargs.get('eta', 0.01))

            if optim == 'adadelta':
                rho = float(kwargs.get('rho', 0.95))
                eps = float(kwargs.get('epsilon', 1e-6))
                logger.info('adadelta(eta=%f, rho=%f, epsilon=%f)', lr, rho, eps)
                self.optimizer = tf.optimizers.Adadelta(lr, rho, eps)
            elif optim == 'adam':
                beta1 = float(kwargs.get('beta1', 0.9))
                beta2 = float(kwargs.get('beta2', 0.999))
                eps = float(kwargs.get('epsilon', 1e-8))
                logger.info('adam(eta=%f beta1=%f, beta2=%f, eps=%f)', lr, beta1, beta2, eps)
                self.optimizer = tf.optimizers.Adam(lr_function, beta1, beta2, eps)
            elif optim == 'adamw':
                import tensorflow_addons as tfa
                wd = float(kwargs.get('weight_decay', 0))
                beta1 = float(kwargs.get('beta1', 0.9))
                beta2 = float(kwargs.get('beta2', 0.999))
                eps = float(kwargs.get('epsilon', 1e-8))
                logger.info('adamw(eta=%f beta1=%f, beta2=%f, eps=%f)', lr, beta1, beta2, eps)
                self.optimizer = tfa.optimizers.AdamW(weight_decay=wd,
                                                      learning_rate=lr_function,
                                                      beta_1=beta1,
                                                      beta_2=beta2,
                                                      epsilon=eps)
            elif optim == 'rmsprop':
                # Get mom again with difference default
                mom = float(kwargs.get('mom', 0.0))
                logger.info('rmsprop(eta=%f, mom=%f)', lr, mom)
                self.optimizer = tf.optimizers.RMSprop(lr_function, momentum=mom)
            elif sgd_mom > 0:
                logger.info('sgd-mom(eta=%f, mom=%f)', lr, sgd_mom)
                self.optimizer = tf.optimizers.SGD(lr_function, sgd_mom)
            else:
                logger.info('sgd(eta=%f)', lr)
                self.optimizer = tf.optimizers.SGD(lr_function)

        logger.info('clip gradients at %s', self.clip)

    def update(self, model, x, y):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, x, y)

        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables), self.global_step)
        return loss_value

    def update_with_hidden(self, model, h, x, y):
        with tf.GradientTape() as tape:
            loss_value, h = self.loss(model, h, x, y)

        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables), self.global_step)
        return loss_value, h

