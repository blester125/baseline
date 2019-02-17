import random
import torch
from baseline.model import register_decoder
from baseline.pytorch.seq2seq.decoders import RNNDecoderWithAttn


@register_decoder(name='scheduled-sample')
class RNNWithSchedulerSampling(RNNDecoderWithAttn):
    def __init__(self, tgt_embeddings, **kwargs):
        self.sample = kwargs.get('sample_rate', 0.5)
        super(RNNWithSchedulerSampling, self).__init__(tgt_embeddings, **kwargs)

    def decode_rnn(self, context_bth, h_i, output_i, dst_tbh, src_mask):
        outputs = []
        for i, dst_i in enumerate(dst_tbh):
            if random.random() < self.sample and i != 0:
                prev_out = self.output(output_i.unsqueeze(0))
                _, prev_out = torch.max(prev_out, dim=-1)
                embed_i = self.tgt_embeddings(prev_out).squeeze(0)
            else:
                embed_i = self.tgt_embeddings(dst_i)

            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_bth, src_mask)
            output_i = self.dropout(output_i)
            outputs.append(output_i)

        outputs_tbh = torch.stack(outputs)
        return outputs_tbh, h_i
