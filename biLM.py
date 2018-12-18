import torch
import torch.nn as nn
from baseline.model import register_model
from baseline.train import register_trainer
from baseline.embeddings import register_embeddings
from baseline.pytorch.lm.model import RNNLanguageModel
from baseline.pytorch.lm.train import LanguageModelTrainerPyTorch
from baseline.pytorch.torchy import SequenceCriterion, pytorch_lstm


@register_trainer(task="lm", name="biLM")
class BiLMTrainerPytorch(LanguageModelTrainerPyTorch):
    def _repack_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self._repack_hidden(v) for v in h)

    def repackage_hidden(self, h):
        f_h, b_h = h
        f_h = self._repack_hidden(f_h)
        if isinstance(f_h, torch.Tensor):
            _, batchsz, _ = f_h.size()
        else:
            _, batchsz, _ = f_h[0].size()
        h = self.model.init_hidden(batchsz)
        return (f_h, h[1])


class BiSequenceCriterion(nn.Module):
    def __init__(self, LossFn=nn.NLLLoss, avg='token'):
        super(BiSequenceCriterion, self).__init__()
        self.fwd = SequenceCriterion(LossFn=LossFn, avg=avg)
        self.bwd = SequenceCriterion(LossFn=LossFn, avg=avg)

    def forward(self, inputs, targets):
        fwd_input, bwd_input = inputs
        fwd_gold, bwd_gold = targets
        fwd_loss = self.fwd(fwd_input, fwd_gold)
        bwd_loss = self.bwd(bwd_input, bwd_gold)
        return fwd_loss + bwd_loss


@register_model(task='lm', name='biLM')
class BiLMRNNLanguageModel(RNNLanguageModel):
    def create_loss(self):
        return BiSequenceCriterion(LossFn=nn.CrossEntropyLoss)

    @staticmethod
    def reverse(x, dim):
        idx = torch.arange(x.size()[dim] - 1, -1, -1).to(x.device)
        return x.index_select(dim, idx)

    def make_input(self, batch_dict):
        example_dict = {}
        for key in self.embeddings.keys():
            example_dict[key] = torch.from_numpy(batch_dict[key])
            if self.gpu:
                example_dict[key] = example_dict[key].cuda()
            example_dict['bwd_{}'.format(key)] = self.reverse(torch.from_numpy(batch_dict[key]), 1)
            if self.gpu:
                example_dict['bwd_{}'.format(key)] = example_dict['bwd_{}'.format(key)].cuda()

        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)
            bwd_y = self.reverse(y, 1)
            if self.gpu is not None:
                y = y.cuda()
                bwd_y = bwd_y.cuda()
            example_dict['y'] = (y, bwd_y)
        return example_dict

    def init_hidden(self, batchsz):
        weight = next(self.parameters()).data
        fwd = (torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()),
               torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()))
        bwd = (torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()),
               torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()))
        return (fwd, bwd)

    def embed(self, inputs):
        fwd_embeddings = []
        bwd_embeddings = []
        for k, embedding in self.embeddings.items():
            fwd_embeddings.append(embedding.encode(inputs[k]))
            bwd_embeddings.append(embedding.encode(inputs['bwd_{}'.format(k)]))
        fwd_embedded = torch.cat(fwd_embeddings, 2)
        bwd_embedded = torch.cat(bwd_embeddings, 2)
        fwd_embedded = self.embed_dropout(fwd_embedded)
        bwd_embedded = self.embed_dropout(bwd_embedded)
        return fwd_embedded, bwd_embedded

    def output(self, x):
        return [self.proj(y) for y in x]

    def init_decode(self, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        vdrop = bool(kwargs.get('variational_dropout', False))
        if vdrop:
            self.rnn_dropout = VariationalDropout(pdrop)
        else:
            self.rnn_dropout = nn.Dropout(pdrop)

        self.fwd_rnn = pytorch_lstm(
            self.dsz, self.hsz, 'lstm', self.layers,
            pdrop, batch_first=True
        )
        self.bwd_rnn = pytorch_lstm(
            self.dsz, self.hsz, 'lstm', self.layers,
            pdrop, batch_first=True
        )

    def decode(self, emb, hidden):
        fwd_emb, bwd_emb = emb
        fwd_hid, bwd_hid = hidden
        fwd_out, fwd_hid = self.fwd_rnn(fwd_emb, fwd_hid)
        bwd_out, bwd_hid = self.bwd_rnn(bwd_emb, bwd_hid)
        fwd_out = self.rnn_dropout(fwd_out).contiguous()
        bwd_out = self.rnn_dropout(bwd_out).contiguous()
        return (fwd_out, bwd_out), (fwd_hid, bwd_hid)


@register_embeddings(name='string')
class ContextualStringEmbeddings(nn.Module, PyTorchEmbeddings):
    def __init__(self):
        super(ContextualStringEmbeddings, self).__init__()

    def encode(self, x):
        pass
