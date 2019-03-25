from baseline.train import register_training_func
from baseline.utils import get_model_file

@register_training_func('classify', 'save')
def save(model, ts, es, vs, **kwargs):
    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))
    model.save_md(model_file)
