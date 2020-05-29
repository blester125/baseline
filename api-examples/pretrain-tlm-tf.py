import logging
import time
import os
from argparse import ArgumentParser
import math
from typing import Tuple
import tempfile
import baseline
from eight_mile.utils import str2bool, write_json
import baseline.tf.embeddings
import baseline.embeddings
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import Average, get_num_gpus_multiworker, read_yaml
from eight_mile.optz import *
from eight_mile.tf.optz import *  #WarmupLinearSchedulerTensorFlow2, CosineDecayTensorFlow2, CompositeLRSchedulerTensorFlow2, EagerOptimizer
from baseline.tf.lm import SET_TRAIN_FLAG, TransformerLanguageModel, TransformerMaskedLanguageModel
import tensorflow as tf
import glob
import json
logger = logging.getLogger(__file__)


"""Pre-train a Transformer model in TensorFlow

Read in preprocessed contexts generated by `preproc-tlm` and use a Transformer Language Model (TLM)
to train them across multiple workers

"""


def get_lr_decay(sched_type, lr, steps_per_epoch, n_epochs, logger, decay_steps=None, decay_rate=None, alpha=None):
    """Copied from transformer_utils.py but that imports torch and we dont want that here

    :param sched_type:
    :param lr:
    :param steps_per_epoch:
    :param n_epochs:
    :param logger:
    :param decay_steps:
    :param decay_rate:
    :param alpha:
    :return:
    """
    if sched_type == 'cosine':
        decay_steps = decay_steps if decay_steps else steps_per_epoch * n_epochs
        alpha = alpha if alpha else 0.
        params = {'decay_steps': decay_steps, 'alpha': alpha}
    else:
        decay_steps = decay_steps if decay_steps else steps_per_epoch
        if not decay_rate:
            if sched_type == 'exponential':
                decay_rate = 0.5
            elif sched_type == 'invtime':
                decay_rate = 1.0
        params = {'decay_steps': decay_steps, 'decay_rate': decay_rate}
    lr_decay = create_lr_scheduler(lr_scheduler_type=sched_type, lr=lr, **params)
    logger.info(f"Using {sched_type} decay learning rate with params {params}.")
    return lr_decay


class Loss:
    def __init__(self, vocab_size, nctx):
        self.vocab_size = vocab_size
        self.nctx = nctx

    def __call__(self, model, features, labels):
        logits, _ = model(features, None)
        loss_mask = tf.cast(labels != 0, tf.float32)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        losses = losses * loss_mask
        losses = tf.reduce_sum(losses)
        non_zero = tf.reduce_sum(loss_mask)
        losses /= non_zero
        return losses


def _parse_json(example):
    j = json.loads(example.numpy())
    return tf.constant(j['x'], dtype=tf.int32), tf.constant(j['y'], dtype=tf.int32)


def decode_json(example):
    return tf.py_function(_parse_json, [example], [tf.int32, tf.int32])


def get_dataset(pattern, num_parallel_reads=1):
    ds = tf.data.TextLineDataset(glob.glob(pattern), num_parallel_reads=num_parallel_reads)
    return ds.map(decode_json)


def get_num_samples(data_dir):
    yml = read_yaml(os.path.join(data_dir, 'md.yml'))
    return yml['num_samples']


def create_distribute_strategy(strategy_name, endpoint=None):
    if strategy_name == 'tpu':
        if endpoint is None:
            endpoint = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=endpoint)
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif strategy_name == 'mirror':
        num_gpus = get_num_gpus_multiworker()
        devices = ['/device:GPU:{}'.format(i) for i in range(num_gpus)]
        strategy = tf.distribute.MirroredStrategy(devices)
    else:
        raise Exception(f"Unsupported strategy {strategy_name}")
    return strategy


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_dir", type=str, required=True, help='Training directory')
    parser.add_argument("--valid_dir", type=str, required=True, help='Validation directory')
    parser.add_argument("--dataset_key", default="tlm",
                        help="dataset key for basedir")
    parser.add_argument("--embed_type", type=str, default='default',
                        choices=["default", "positional", "learned-positional"],
                        help="register label of the embeddings")
    parser.add_argument("--d_model", type=int, default=410, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2100, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=2, help="Number valid workers")
    parser.add_argument("--distribute", type=str, default="mirror", choices=["mirror", "tpu"])
    parser.add_argument("--tpu_ep", type=str, help="The TPU endpoint if using `distribute=tpu`")
    parser.add_argument("--nctx", type=int, default=128, help="Max input length")
    parser.add_argument("--pattern", default='*.txt', help="Glob pattern for data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--optim", default="adam", type=str, help="Optimizer to use (defaults to adam)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints (step|epoch)")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--lr_scheduler", type=str, help="The type of learning rate decay scheduler", default='cosine')
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--mlm", type=str2bool, default=True, help="Use Masked Language Model (MLM) objective")
    parser.add_argument("--saves_per_epoch", type=int, default=100, help="The number of checkpoints to save per epoch")
    parser.add_argument('--rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')
    parser.add_argument("--strategy", help="Training strategy, defaults to `mirror`", choices=["mirror"])

    args = parser.parse_args()
    SET_TRAIN_FLAG(True)

    if args.basedir is None:
        args.basedir = 'lm-{}-bpe-{}'.format(args.dataset_key, os.getpid())
    logging.basicConfig(level=logging.INFO)

    strategy = create_distribute_strategy(args.distribute, args.tpu_ep)
    num_replicas = strategy.num_replicas_in_sync
    logger.info(f"Using {num_replicas} replicas in this job.")
    global_step = 1
    vectorizer = BPEVectorizer1D(model_file=args.subword_model_file, vocab_file=args.subword_vocab_file, mxlen=args.nctx)
    vocab = {'x': vectorizer.vocab}
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab['x'],
                                                       preserve_vocab_indices=True,
                                                       embed_type=args.embed_type)
    vocabs = preproc_data['vocab']
    vocab_size = max(vocabs.values())

    def dataset_train_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(args.batch_size)
        ds = get_dataset(os.path.join(args.train_dir, args.pattern), args.num_train_workers).batch(batch_size)
        return ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    train_loader = strategy.experimental_distribute_datasets_from_function(dataset_train_fn)

    def dataset_test_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(args.batch_size)
        ds = get_dataset(os.path.join(args.valid_dir, args.pattern), args.num_train_workers).batch(batch_size)
        return ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    valid_loader = strategy.experimental_distribute_datasets_from_function(dataset_test_fn)

    num_train_samples = get_num_samples(args.train_dir)
    num_valid_samples = get_num_samples(args.valid_dir)
    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    embeddings = {'x': preproc_data['embeddings']}
    logger.info("Loaded embeddings")

    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)
    if len(args.rpr_k) == 0 or args.rpr_k[0] < 1:
        rpr_k = None
    elif len(args.rpr_k) == 1:
        rpr_k = args.rpr_k[0]
    else:
        rpr_k = args.rpr_k

    TLM = TransformerMaskedLanguageModel if args.mlm else TransformerLanguageModel
    model = TLM.create(embeddings,
                       hsz=args.d_model,
                       d_ff=args.d_ff,
                       tie_weights=True,
                       dropout=args.dropout,
                       gpu=False,
                       num_heads=args.num_heads,
                       layers=args.num_layers,
                       rpr_k=rpr_k,
                       d_k=args.d_k,
                       src_keys=['x'], tgt_key='x')

    loss_function = Loss(vocab_size, args.nctx)

    logger.info("Loaded model and loss")
    steps_per_epoch = math.ceil(num_train_samples / args.batch_size)
    steps_per_valid_epoch = math.ceil(num_valid_samples / args.batch_size)
    update_on = steps_per_epoch // args.saves_per_epoch
    report_on = max(10, update_on) // 10
    logger.info(f"Steps per epoch per GPU: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")

    lr_decay = get_lr_decay(args.lr_scheduler, args.lr, steps_per_epoch, args.epochs, logger,
                            decay_steps=args.lr_decay_steps, decay_rate=args.lr_decay_rate, alpha=args.lr_alpha)
    linear_warmup = WarmupLinearSchedulerTensorFlow(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRSchedulerTensorFlow(linear_warmup, lr_decay, lr=args.lr)
    optimizer = EagerOptimizer(loss_function, global_step=global_step, lr=args.lr, optim=args.optim,
                               learning_rate_decay_fn=lr_sched, weight_decay=args.weight_decay, clip=args.clip)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer.optimizer, model=model)
    pid = os.getpid()
    checkpoint_dir = f"./tf-{args.dataset_key}-{pid}"
    #model_base = os.path.join(args.basedir, 'checkpoint-step')

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory=checkpoint_dir,
                                                    max_to_keep=5)

    if args.restart_from:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    def _replicated_train_step(inputs):
        """This runs on a single replica"""
        x, y = inputs
        per_replica_loss = optimizer.update(model, {'x': x}, y, num_replicas)
        return per_replica_loss

    @tf.function
    def _distributed_train_step(inputs: Tuple[tf.Tensor, tf.Tensor]):
        """Runs across multiple replicas and aggregates the results.

        :param inputs:
        :return:
        """
        per_replica_loss = strategy.experimental_run_v2(_replicated_train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    valid_loss_function = Loss(vocab_size, args.nctx)
    def _replicated_test_step(inputs):
        """This runs on a single replica"""
        x, y = inputs
        per_replica_loss = valid_loss_function(model, {'x': x}, y) / num_replicas
        return per_replica_loss

    @tf.function
    def _distributed_test_step(inputs: Tuple[tf.Tensor, tf.Tensor]):
        """Runs across multiple replicas and aggregates the results.

        :param inputs:
        :return:
        """
        per_replica_loss = strategy.experimental_run_v2(_replicated_test_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    # This is the training loop
    steps = global_step
    start_epoch = 0

    with strategy.scope():

        SET_TRAIN_FLAG(True)
        for epoch in range(start_epoch, args.epochs):
            avg_loss = Average('average_train_loss')
            metrics = {}
            start = time.time()
            train_iter = iter(train_loader)
            for i in range(steps_per_epoch):
                steps += 1
                loss = _distributed_train_step(next(train_iter))
                avg_loss.update(loss.numpy().item())
                if (i + 1) % report_on == 0:
                    logging.info(avg_loss)
                if (i + 1) % update_on == 0:
                    elapsed = (time.time() - start)/60
                    logging.info('elapsed time this epoch %d min', elapsed)
                    logging.info('elapsed step time %f steps/min', i/elapsed)
                    checkpoint_manager.save()

            # How much time elapsed in minutes
            elapsed = (time.time() - start)/60
            train_token_loss = avg_loss.avg
            # This is the average training token-level loss across all machines
            # This is the token-level training perplexity
            train_token_ppl = math.exp(train_token_loss)
            metrics['train_elapsed_min'] = elapsed
            metrics['average_train_loss'] = train_token_loss
            metrics['train_ppl'] = train_token_ppl
            avg_valid_loss = Average('average_valid_loss')
            start = time.time()
            SET_TRAIN_FLAG(False)
            valid_iter = iter(valid_loader)
            for i in range(steps_per_valid_epoch):
                valid_loss = _distributed_test_step(next(valid_iter))
                avg_valid_loss.update(valid_loss.numpy().item())

            valid_token_loss = avg_valid_loss.avg
            valid_token_ppl = math.exp(valid_token_loss)

            elapsed = (time.time() - start)/60
            metrics['valid_elapsed_min'] = elapsed
            metrics['average_valid_loss'] = valid_token_loss
            metrics['average_valid_word_ppl'] = valid_token_ppl
            logger.info(metrics)


if __name__ == "__main__":
    train()

