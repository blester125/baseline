#!/usr/bin/env python3


import torch
import torch.nn as nn
from eight_mile.pytorch.layers import repeat_batch, MaxPool1D, MeanPool1D
from baseline.pytorch.seq2seq.model import EncoderDecoderModelBase, Seq2SeqModel
from baseline.pytorch.torchy import *
from baseline.model import register_model, register_decoder
from baseline.pytorch.seq2seq.decoders import RNNDecoderWithAttn


class NBestDecoderMixin(nn.Module):
    """This is a class for collapsing the NBest during the decoder step."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nbest_agg = self.init_nbest_agg(None, **kwargs)

    def init_nbest_agg(self, input_dim: int, **kwargs) -> BaseLayer:
        # This should probably be done with subclassing that overrides this class but
        # this would cause a huge explosion of classes for the combo of aggs and poolers
        agg_type = kwargs.get("agg_type", "max")
        if agg_type == "max":
            return MaxPool1D(input_dim)
        elif agg_type == "mean":
            return MeanPool1D(input_dim)
        # TODO: Add other pooling mechanisms like attention or hard selection of a single hypothesis
        elif agg_type == "":
            pass
        else:
            raise ValueError(f"Unknown NBest aggregation function, got: {agg_type}")

    def forward(self, encoder_outputs, dst, inputs):
        src_mask = encoder_outputs.src_mask
        h_i, output_i, context_bth = self.arc_policy(encoder_outputs, self.hsz)
        output_tbh, _ = self.decode_rnn(context_bth, h_i, output_i, dst.transpose(0, 1), src_mask)

        nbest_lengths = inputs['nbest_lengths']
        perm_idx = inputs['perm_idx']

        output_bth = output_tbh.permute(1, 0, 2).contiguous()
        output_bth = unsort_batch(output_bth, perm_idx)
        B = nbest_lengths.size(0)
        _, T, H = output_bth.shape
        output_bnth = output_bth.view(B, -1, T, H)
        output_bnx = output_bnth.view(B, -1, T * H)
        output_bx = self.nbest_agg((output_bnx, nbest_lengths))
        output_bth = output_bx.view(B, T, H)
        output_tbh = output_bth.permute(1, 0, 2).contiguous()

        pred = self.output(output_tbh)
        return pred.transpose(0, 1).contiguous()


class NBestEncoderDecoderMixin(EncoderDecoderModelBase):
    def input_tensor(self, key, batch_dict, perm_idx, max_n, numpy_to_tensor=False):
        tensor = batch_dict[key]
        if numpy_to_tensor:
            tensor = torch.from_numpy(tensor)
        tensor = tensor[:, :max_n].contiguous()

        B, N, *rest = tensor.size()
        tensor = tensor.view(tuple([B * N] + rest))

        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        if self.gpu:
            tensor = tensor.cuda()
        return tensor

    def forward(self, input: Dict[str, torch.Tensor]):
        src_len = input['src_len']
        encoder_outputs = self.encode(input, src_len)
        output = self.decode(encoder_outputs, input['dst'], input)
        return output

    def decode(self, encoder_outputs, dst, inputs):
        return self.decoder(encoder_outputs, dst, inputs)


    def make_input(self, batch_dict, perm=False, numpy_to_tensor=False):
        example = {}

        nbest_lengths = batch_dict[f"{self.src_lengths_key}_nbest"]
        if numpy_to_tensor:
            nbest_lengths = torch.from_numpy(nbest_lengths)
        if self.gpu:
            nbest_lengths = nbest_lengths.cuda()
        example['nbest_lengths'] = nbest_lengths
        max_n = torch.max(nbest_lengths)

        lengths = batch_dict[self.src_lengths_key]
        if numpy_to_tensor:
            lengths = torch.from_numpy(lengths)
        lengths = lengths[:, :max_n].contiguous()
        B, N = lengths.size()
        lengths = lengths.view(B * N)
        lengths, perm_idx = lengths.sort(0, descending=True)
        if self.gpu:
            lengths = lengths.cuda()
        example['src_len'] = lengths
        example['perm_idx'] = perm_idx

        for key in self.src_embeddings.keys():
            example[key] = self.input_tensor(key, batch_dict, perm_idx, max_n=max_n, numpy_to_tensor=numpy_to_tensor)

        if 'tgt' in batch_dict:
            tgt = batch_dict['tgt']
            if numpy_to_tensor:
                tgt = torch.from_numpy(tgt)

            dst = tgt[:, :-1]
            tgt = tgt[:, 1:]

            dst = repeat_batch(dst, max_n)
            dst = dst[perm_idx]

            if self.gpu:
                dst = dst.cuda()
                tgt = tgt.cuda()

            example['dst'] = dst
            example['tgt'] = tgt
        if perm:
            return example, perm_idx
        return example


@register_model(task="seq2seq", name="nbest-attn")
class NBestSeq2SeqModel(NBestEncoderDecoderMixin, Seq2SeqModel):
    pass


@register_decoder(name="nbest-attn")
class NBestRNNDecoderWtihAttn(NBestDecoderMixin, RNNDecoderWithAttn):
    pass
