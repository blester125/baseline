#!/usr/bin/env python3

from eight_mile.pytorch.layers import MaxPool1D, unsort_batch
from baseline.model import register_model
from baseline.pytorch.torchy import *
from baseline.pytorch.classify.model import EmbedPoolStackModel, ConvModel, ClassifierModelBase, FineTuneModelClassifier


class NBestMixin(ClassifierModelBase):

    def init_nbest_agg(self, input_dim: int, **kwargs) -> BaseLayer:
        # This should probably be done with subclassing that overrides this class but
        # this would cause a huge explosion of classes for the combo of aggs and poolers
        agg_type = kwargs.get("agg_type", "max")
        if agg_type == "max":
            return MaxPool1D(input_dim)
        elif agg_type == "mean":
            return MeanPool1D(input_dim)
        elif agg_type == "":
            pass
        else:
            raise ValueError(f"Unknown NBest aggregation function, got: {agg_type}")

    def make_input(self, batch_dict, perm=False, numpy_to_tensor=False):
        example_dict = dict({})
        perm_idx = None

        nbest_lengths = batch_dict[f"{self.lengths_key}_nbest"]
        if numpy_to_tensor:
            nbest_lengths = torch.from_numpy(nbest_lengths)
        if self.gpu:
            nbest_lengths = nbest_lengths.cuda()
        example_dict['nbest_lengths'] = nbest_lengths
        max_n = torch.max(nbest_lengths)

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            lengths = batch_dict[self.lengths_key]
            if numpy_to_tensor:
                lengths = torch.from_numpy(lengths)
            lengths = lengths[:, :max_n].contiguous()
            B, N = lengths.shape
            lengths = lengths.view(B * N)
            lengths, perm_idx = lengths.sort(0, descending=True)
            if self.gpu:
                lengths = lengths.cuda()
            example_dict['lengths'] = lengths

        for key in self.embeddings.keys():
            tensor = batch_dict[key]
            if numpy_to_tensor:
                tensor = torch.from_numpy(tensor)
            tensor = tensor[:, :max_n].contiguous()
            B, N, *rest = tensor.shape
            tensor = tensor.view(tuple([B * N] + rest))
            if perm_idx is not None:
                tensor = tensor[perm_idx]
            if self.gpu:
                tensor = tensor.cuda()
            example_dict[key] = tensor

        y = batch_dict.get('y')
        if y is not None:
            if numpy_to_tensor:
                y = torch.from_numpy(y)
            if self.gpu:
                y = y.cuda()
            example_dict['y'] = y

        example_dict['perm_idx'] = perm_idx

        if perm:
            return example_dict, perm_idx
        return example_dict


class NBestEmbedPoolStackMixin(NBestMixin):

    def create_layers(self, embeddings: Dict[str, BaseLayer], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.pool_model = self.init_pool(self.embeddings.output_dim, **kwargs)
        self.stack_model = self.init_stacked(self.pool_model.output_dim, **kwargs)
        self.nbest_agg = self.init_nbest_agg(self.stack_model.output_dim, **kwargs)
        self.output_layer = self.init_output(self.nbest_agg.output_dim, **kwargs)

    def forward(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Forward execution of the model.  Sub-classes typically shouldnt need to override

        :param inputs: An input dictionary containing the features and the primary key length
        :return: A tensor
        """
        lengths = inputs.get("lengths")
        n_lengths = inputs['nbest_lengths']
        perm_idx = inputs['perm_idx']

        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled)

        stacked = unsort_batch(stacked, perm_idx)
        B = n_lengths.size(0)
        stacked = stacked.view(B, -1, self.stack_model.output_dim)

        agged = self.nbest_agg((stacked, n_lengths))
        return self.output_layer(agged)


class NBestFineTuneMixin(NBestMixin):

    def create_layers(self, embeddings: Dict[str, BaseLayer], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.stack_model = self.init_stacked(self.embeddings.output_dim, **kwargs)
        self.nbest_agg = self.init_nbest_agg(self.stack_model.output_dim, **kwargs)
        self.output_layer = self.init_output(self.nbest_agg.output_dim, **kwargs)

    def forward(self, inputs):
        n_lengths = inputs['nbest_lengths']
        perm_idx = inputs['perm_idx']

        base_layers = self.embeddings(inputs)
        stacked = self.stack_model(base_layers)

        stacked = unsort_batch(stacked, perm_idx)
        B = n_lengths.size(0)
        stacked = stacked.view(B, -1, self.stack_model.output_dim)

        agged = self.nbest_agg((stacked, n_lengths))
        return self.output_layer(agged)


@register_model(task="classify", name="nbest-conv")
class NBestConvModel(NBestEmbedPoolStackMixin, ConvModel):
    pass


@register_model(task="classify", name="nbest-fine-tune")
class NBestFineTuneModel(NBestFineTuneMixin, FineTuneModelClassifier):
    pass
