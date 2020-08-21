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
    def predict(self, batch_dict, **kwargs):
        """Predict based on the batch.

        If `make_input` is True then run make_input on the batch_dict.
        This is false for being used during dev eval where the inputs
        are already transformed.
        """
        self.eval()
        make = kwargs.get('make_input', True)
        if make:
            numpy_to_tensor = bool(kwargs.get('numpy_to_tensor', True))
            inputs, perm_idx = self.make_input(batch_dict, perm=True, numpy_to_tensor=numpy_to_tensor)
        else:
            inputs = batch_dict
        encoder_outputs = self.encode(inputs, inputs['src_len'])
        outs, lengths, scores = self.decoder._greedy_search(encoder_outputs, inputs=inputs, **kwargs)
        if make:
            lengths = unsort_batch(lengths, perm_idx)
            scores = unsort_batch(scores, perm_idx)
        return outs.cpu().numpy()


@register_decoder(name="nbest-attn")
class NBestRNNDecoderWtihAttn(NBestDecoderMixin, RNNDecoderWithAttn):
    def _greedy_search(self, encoder_output, **kwargs):
        """Decode a sentence by taking the hightest scoring token at each timestep.

        In the past we have just used a beam size of 1 instead of a greedy search because
        they took about the same time to run. I have added this function back because it
        is easier to debug and can help finding where different problems in the output are.

        :param encoder_output: `EncoderOutput` The output of the encoder, it should be
            in the batch first format.
        """
        bsz = encoder_output.output.shape[0]
        device = encoder_output.output.device
        mxlen = int(kwargs.get("mxlen", 100))
        nbest_lengths = kwargs['inputs']['nbest_lengths']
        perm_idx = kwargs['inputs']['perm_idx']
        N = torch.max(nbest_lengths)
        B = nbest_lengths.size(0)
        with torch.no_grad():
            src_mask = encoder_output.src_mask  # [B, T]
            # h_i = Tuple[[B * N, H], [B * N, H]]
            # dec_out = [B * N, H]
            # context = [B * N, T, H]
            h_i, dec_out, context = self.arc_policy(encoder_output, self.hsz)
            # The internal `decode_rnn` actually takes time first so to that.
            last = torch.full((1, B), Offsets.GO, dtype=torch.long, device=device)
            outputs = [last]
            last = repeat_batch(last, N, dim=1)
            for i in range(mxlen - 1):
                # Take a step with the RNN
                # dec_out = [1, B, H]
                # hi = Tuple[[B, H], [B, H]]
                dec_out, h_i = self.decode_rnn(context, h_i, dec_out, last, src_mask) # [1, B * N, H]
                dec = unsort_batch(dec_out.squeeze(0), perm_idx)
                dec = dec.view(B, N, -1)  # [B, N, H]
                dec = self.nbest_agg((dec, nbest_lengths))  # [B, H]
                dec = dec.unsqueeze(0)
                # Project to vocab size
                probs = self.output(dec)  # [1, B, V]
                # Get the best scoring token for each timestep in the batch
                selected = torch.argmax(probs, dim=-1)
                outputs.append(selected)
                selected = repeat_batch(selected, N, dim=1)  # [1, B * N]
                selected = selected.view(-1)[perm_idx].view(1, -1)
                last = selected
                # Convert the last step of the decoder output into a format we can consume [B, H]
                dec_out = dec_out.squeeze(0)
            # Combine all the [1, B] outputs into a [T, B] matrix
            outputs = torch.cat(outputs, dim=0)
            # Convert to [B, T]
            outputs = outputs.transpose(0, 1).contiguous()
            # Add a fake beam dimension of size 1
            outputs = outputs.unsqueeze(1)
            # This is mostly for testing so just return zero for lengths and scores.
            return outputs, torch.zeros(bsz), torch.zeros(bsz)
