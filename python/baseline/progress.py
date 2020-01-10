import eight_mile.progress
from eight_mile.progress import *

__all__ = []
__all__.extend(eight_mile.progress.__all__)


MEAD_LAYERS_PROGRESS = {}


@export
@optional_params
def register_progress(cls, name=None):
    return register(cls, MEAD_LAYERS_PROGRESS, name, 'progress')


@export
def create_progress_bar(steps, name='default', **kwargs):
    return MEAD_LAYERS_PROGRESS[name](steps, **kwargs)
