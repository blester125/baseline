import tensorflow as tf
from eight_mile.optz import register_lr_scheduler, create_lr_scheduler, WarmupLearningRateScheduler, LearningRateScheduler, CompositeLRScheduler
from eight_mile.utils import get_version
from baseline.optz import register_lr_scheduler


if get_version(tf) < 2:

    @register_lr_scheduler('default')
    class ConstantSchedulerTensorFlow1:
        def __init__(self, **kwargs):
            pass

        def __call__(self, lr, global_step):
            return tf.identity(lr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('warmup_linear')
    class WarmupLinearSchedulerTensorFlow1(WarmupLearningRateScheduler):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, lr, global_step):
            return tf.identity(tf.minimum(1.0, tf.cast(global_step / self.warmup_steps, dtype=tf.float32)) * lr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"

    @register_lr_scheduler('clr')
    class CyclicLRSchedulerTensorFlow1:
        def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
            super().__init__()
            self.max_lr = max_lr
            self.decay_steps = decay_steps

        def __call__(self, lr, global_step):
            gs_f = tf.cast(global_step, tf.float32)
            cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
            x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
            clr = lr + (self.max_lr - lr) * tf.maximum(0., 1. - x)
            return tf.identity(clr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('sgdr')
    class SGDRSchedulerTensorFlow1:
        def __init__(self, first_decay_steps=1000, **kwargs):
            super().__init__()
            self.first_decay_steps = first_decay_steps

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.cosine_decay_restarts(lr, global_step, first_decay_steps=self.first_decay_steps), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('composite')
    class CompositeLRSchedulerTensorFlow1:
        def __init__(self, warm=None, rest=None, **kwargs):
            self.warm = warm
            self.rest = rest

        def __call__(self, lr, global_step):
            warm_tensor = self.warm(lr, global_step)

            def call_warm():
                return warm_tensor

            rest_step = tf.subtract(global_step, tf.compat.v1.constant(self.warm.warmup_steps, dtype=global_step.dtype))
            rest_tensor = self.rest(lr, rest_step)

            def call_rest():
                return rest_tensor

            return tf.identity(tf.cond(
                global_step < self.warm.warmup_steps,
                call_warm, call_rest
            ), name='lr')

        def __str__(self):
            return "LRScheduler({}, {})".format(self.warm, self.rest)

    @register_lr_scheduler('piecewise')
    class PiecewiseDecaySchedulerTensorFlow(object):

        def __init__(self, boundaries=None, values=None, **kwargs):
            super(PiecewiseDecaySchedulerTensorFlow, self).__init__()
            self.boundaries = boundaries
            self.values = values

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.piecewise_constant(global_step, self.boundaries, self.values), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('invtime')
    class InverseTimeDecaySchedulerTensorFlow:
        def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
            super().__init__()
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
            self.staircase = staircase

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.inverse_time_decay(lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('exponential')
    class ExponentialDecaySchedulerTensorFlow(object):
        def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
            self.staircase = staircase

        def __call__(self, lr, global_step):
            return tf.identity(tf.train.exponential_decay(lr, global_step, self.decay_steps, self.decay_rate, staircase=self.staircase), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('zaremba')
    class ZarembaDecaySchedulerTensorFlow1(PiecewiseDecaySchedulerTensorFlow):
        """Utility only, just to simplify the JSON"""
        def __init__(self, boundaries=None, decay_rate=None, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            values = [lr/(float(decay_rate)**i) for i in range(len(boundaries)+1)]
            super().__init__(boundaries=boundaries, values=values)

        def __str__(self):
            return type(self).__name__ + "()"


else:

    @register_lr_scheduler('default')
    class ConstantSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, global_step):
            return self.lr

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('warmup_linear')
    class WarmupLinearSchedulerTensorFlow2(WarmupLearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            kwargs['lr'] = lr
            super().__init__(**kwargs)

        def __call__(self, global_step):
            return tf.minimum(1.0, global_step / float(self.warmup_steps)) * self.lr

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('clr')
    class CyclicLRSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            kwargs['lr'] = lr
            super().__init__(**kwargs)
            self.max_lr = max_lr
            self.decay_steps = decay_steps

        def __call__(self, global_step):
            gs_f = tf.cast(global_step, tf.float32)
            cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
            x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
            clr = self.lr + (self.max_lr - self.lr) * tf.maximum(0., 1. - x)
            return tf.identity(clr, name='lr')

        def __str__(self):
            return type(self).__name__ + "()"

    @register_lr_scheduler('sgdr')
    class SGDRSchedulerTensorFlow2(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, first_decay_steps=1000, **kwargs):
            super().__init__(**kwargs)
            self.first_decay_steps = first_decay_steps

        def __call__(self, global_step):
            return tf.identity(tf.compat.v1.train.cosine_decay_restarts(self.lr, global_step, first_decay_steps=self.first_decay_steps), name='lr')

        def __str__(self):
            return type(self).__name__ + "()"


    @register_lr_scheduler('zaremba')
    class ZarembaDecaySchedulerTensorFlow2(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
        """Utility only, just to simplify the JSON"""
        def __init__(self, boundaries=None, decay_rate=None, **kwargs):
            lr = float(kwargs.get('lr', kwargs.get('eta', 1.0)))
            values = [lr/(float(decay_rate)**i) for i in range(len(boundaries)+1)]
            super().__init__(boundaries, values, kwargs.get('name'))


    @register_lr_scheduler('composite')
    class CompositeLRSchedulerTensorFlow2(CompositeLRScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
        pass

    @register_lr_scheduler('piecewise')
    class PiecewiseConstantDecayTensorFlow2(tf.keras.optimizers.schedules.PiecewiseConstantDecay):

        def __init__(self, boundaries, values, **kwargs):
            super().__init__(boundaries, values)

    @register_lr_scheduler('invtime')
    class InverseTimeDecayTensorFlow2(tf.keras.optimizers.schedules.InverseTimeDecay):

        def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
            lr = kwargs.get('lr', kwargs.get('eta', 0.01))
            super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get('name'))

    @register_lr_scheduler('exponential')
    class ExponentialDecayTensorFlow2(tf.keras.optimizers.schedules.ExponentialDecay):
        def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
            lr = kwargs.get('lr', kwargs.get('eta', 0.01))
            super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get('name'))
