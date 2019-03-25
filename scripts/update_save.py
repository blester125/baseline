import re
import logging
import argparse
from copy import deepcopy
from itertools import chain
from baseline.utils import read_config_stream, normalize_backend, unzip_files
import mead
from mead.utils import convert_path, parse_extra_args, configure_logger

logger = logging.getLogger('mead')


def find_pid(base, file_name):
    pid_re = re.compile("%s-(\d{1,5})(?:\.zip)?" % base)
    m = pid_re.match(os.path.basename(file_name))
    if m is not None:
        return m.groups()[0]


def find_file(base, pid):
    if os.path.exists("{}-{}.zip".format(base, pid)):
        return "{}-{}.zip".format(base, pid)
    return "{}-{}".format(base, pid)


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='JSON Configuration for an experiment', type=convert_path, default="$MEAD_CONFIG")
    parser.add_argument('--model', help='The model you are trying to update')

    parser.add_argument('--settings', help='JSON Configuration for mead', default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--backend', help='The deep learning backend to use')
    args, reporting_args = parser.parse_known_args()

    args.logging = read_config_stream(args.logging)
    configure_logger(args.logging)

    config_params = read_config_stream(args.config)
    try:
        args.settings = read_config_stream(args.settings)
    except:
        logger.warning('Warning: no mead-settings file was found at [{}]'.format(args.settings))
        args.settings = {}

    config_params['modules'] = config_params.get('modules', []) + ['save']
    print(config_params)
    config_params['train']['fit_func'] = 'save'

    args.datasets = read_config_stream(args.datasets)
    args.embeddings = read_config_stream(args.embeddings)

    if args.backend is not None:
        config_params['backend'] = normalize_backend(args.backend)

    task_name = config_params.get('task', 'classify') if args.task is None else args.task
    logger.info('Task: [{}]'.format(task_name))
    task = mead.Task.get_task_specific(task_name, args.settings)
    task.read_config(config_params, args.datasets, reporting_args=reporting_args, config_file=deepcopy(config_params))
    task.initialize(args.embeddings)
    task.train()

    base = config_params['basedir']

    old_pid = find_pid(base, args.model)
    new_pid = os.getpid()
    new_file = find_file(base, new_pid)

    print(args.model)
    print(new_file)


if __name__ == "__main__":
    main()
