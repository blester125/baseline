import eight_mile.optz
from eight_mile.optz import *
from eight_mile.utils import exporter


__all__ = []
__all__.extend(eight_mile.optz.__all__)
MEAD_LAYERS_LR_SCHEDULERS = {}
export = exporter(__all__)


@export
@optional_params
def register_lr_scheduler(cls, name=None):
    return register(cls, MEAD_LAYERS_LR_SCHEDULERS, name, 'lr_scheduler')


@export
def create_lr_scheduler(**kwargs):
    """Create a learning rate scheduler.

    :Keyword Arguments:
      * *lr_scheduler_type* `str` or `list` The name of the learning rate scheduler
          if list then the first scheduler should be a warmup scheduler.
    """
    sched_type = kwargs.get('lr_scheduler_type')
    if sched_type is None:
        return None
    sched_type = listify(sched_type)
    if len(sched_type) == 2:
        warm = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[0])(**kwargs)
        assert isinstance(warm, WarmupLearningRateScheduler), "First LR Scheduler must be a warmup scheduler."
        rest = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[1])(**kwargs)
        return MEAD_LAYERS_LR_SCHEDULERS.get('composite')(warm=warm, rest=rest, **kwargs)
    Constructor = MEAD_LAYERS_LR_SCHEDULERS.get(sched_type[0])
    return Constructor(**kwargs)
