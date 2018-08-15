import tensorflow as tf
from baseline.tf.tfy import stacked_lstm
from baseline.tf.classify.model import WordClassifierBase

class LSTMModel(WordClassifierBase):
    def __init__(self):
        super(LSTMModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if isinstance(hsz, list):
            hsz = hsz[0]

        rnntype = kwargs.get('rnntype', 'lstm')
        nlayers = int(kwargs.get('layers', 1))

        learn_init = bool(kwargs.get('learn_init', False))

        if rnntype == 'blstm':
            share_state = bool(kwargs.get('share_init', False))
            fw_state, bw_state = None, None
            if learn_init:
                batch_size = tf.shape(word_embeddings)[0]
                fw_state = get_state_variable(nlayers, hsz, init, batch_size, scope="fw_state")
                if share_state:
                    bw_state = fw_state
                else:
                    bw_state = get_state_variable(nlayers, hsz, init, batch_size, scope="bw_state")
            rnnfwd = stacked_lstm(hsz, self.pkeep, nlayers)
            rnnbwd = stacked_lstm(hsz, self.pkeep, nlayers)
            _, (fw_final_state, bw_final_state) = tf.nn.bidirectional_dynamic_rnn(
                rnnfwd, rnnbwd,
                word_embeddings,
                sequence_length=self.lengths,
                initial_state_fw=fw_state,
                initial_state_bw=bw_state,
                dtype=tf.float32
            )
            output_state = fw_final_state[-1].h + bw_final_state[-1].h
            out_hsz = hsz

        else:
            state = None
            if learn_init:
                state = get_state_variable(nlayers, hsz, init, tf.shape(word_embeddings)[0])
            rnn = stacked_lstm(hsz, self.pkeep, nlayers)
            _, output_state = tf.nn.dynamic_rnn(
                rnn, word_embeddings, sequence_length=self.lengths, dtype=tf.float32, initial_state=state
            )
            output_state = output_state[-1].h
            out_hsz = hsz

        combine = tf.reshape(output_state, [-1, out_hsz])
        return combine


def get_state_variable(nlayers, hsz, init, batch_size, scope="initial_state"):
    with tf.variable_scope(scope):
        cs = []
        hs = []
        for i in range(nlayers):
            cs.append(tf.get_variable("c_{}".format(i), shape=[1, hsz], initializer=init))
            hs.append(tf.get_variable("h_{}".format(i), shape=[1, hsz], initializer=init))
        state = []
        for c, h in zip(cs, hs):
            c = tf.tile(c, [batch_size, 1])
            h = tf.tile(h, [batch_size, 1])
            state.append(tf.contrib.rnn.LSTMStateTuple(c, h))
        return tuple(state)


def create_model(embeddings, labels, **kwargs):
    return LSTMModel.create(embeddings, labels, **kwargs)

def load_model(model_file, **kwargs):
    return LSTMModel.load(model_file, **kwargs)
