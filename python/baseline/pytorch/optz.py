from eight_mile.optz import (
    register_lr_scheduler,
    create_lr_scheduler,
    ConstantScheduler,
    WarmupLinearScheduler,
    CyclicLRScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    CosineDecayScheduler,
    InverseTimeDecayScheduler,
    ExponentialDecayScheduler,
    CompositeLRScheduler,
)


@register_lr_scheduler(name='default')
class ConstantSchedulerPyTorch(ConstantScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@register_lr_scheduler(name='warmup_linear')
class WarmupLinearSchedulerPyTorch(WarmupLinearScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name='clr')
class CyclicLRSchedulerPyTorch(CyclicLRScheduler):

    def __init__(self, *args, **kwargs):
        super().__init(*args, **kwargs)


@register_lr_scheduler(name='piecewise')
class PiecewiseDecaySchedulerPyTorch(PiecewiseDecayScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name='zaremba')
class ZarembaDecaySchedulerPyTorch(ZarembaDecayScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name='cosine')
class CosineDecaySchedulerPyTorch(CosineDecayScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name='invtime')
class InverseTimeDecaySchedulerPytorch(InverseTimeDecayScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name='exponential')
class ExponentialDecaySchedulerPyTorch(ExponentialDecayScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_lr_scheduler(name='composite')
class CompositeLRSchedulerPyTorch(CompositeLRScheduler):
    pass
