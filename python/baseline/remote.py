from six.moves.http_client import HTTPConnection
from six.moves.urllib.parse import urlparse

import json
import numpy as np
from baseline.utils import (
    import_user_module, export, optional_params, register, listify
)


__all__ = []
exporter = export(__all__)

BASELINE_REMOTES = {}

@exporter
@optional_params
def register_remote(cls, name=None):
    """Register a class as a plug-in"""
    if name is None:
        name = cls.__name__
    names = listify(name)
    for alias in names:
        if alias in BASELINE_REMOTES:
            raise Exception('Error: attempt to re-define previously registered hander {} (old: {}, new: {}) in registry'.format(alias, BASELINE_REMOTES, cls))
        BASELINE_REMOTES[alias] = cls
    return cls


@exporter
def create_remote(exporter_type, **kwargs):
    remote = BASELINE_REMOTES[exporter_type]
    return remote(**kwargs)


def verify_example(examples, keys):
    missing_keys = [k for k in keys if k not in examples]
    if missing_keys:
        raise ValueError("Data should have keys: {}\n {} are missing.".format(keys, missing_keys))


class RemoteModel(object):
    def __init__(
            self,
            remote,
            name, signature,
            labels=None,
            beam=None,
            lengths_key=None,
            inputs=None,
            version=None,
            return_labels=None,
    ):
        """A remote model where the actual inference is done on a server.

        :param remote: The remote endpoint
        :param name:  The name of the model
        :param signature: The model signature
        :param labels: The labels (defaults to None)
        :param beam: The beam width (defaults to None)
        :param lengths_key: Which key is used for the length of the input
            vector (defaults to None)
        :param inputs: The inputs (defaults to empty list)
        :param version: The model version (defaults to None)
        :param return_labels: Whether the remote model returns class indices or
            the class labels directly. This depends on the `return_labels`
            parameter in exporters
        """
        inputs = [] if inputs is None else inputs
        self.remote = remote
        self.name = name
        self.signature = signature
        self.lengths_key = lengths_key
        self.input_keys = inputs
        self.beam = beam
        self.labels = labels
        self.version = version
        self.return_labels = return_labels

    def get_labels(self):
        """Return the model's labels

        :return: The model's labels
        """
        return self.labels

    def predict(self, examples, **kwargs):
        """Run inference on examples."""
        pass

    def deserialize_response(self, examples, predict_response):
        """Convert the response into a standard format."""
        pass

    def create_request(self, examples):
        """Convert the examples into the format expected by the server."""
        pass


@exporter
class RemoteModelREST(RemoteModel):

    def __init__(
            self,
            remote,
            name, signature,
            labels=None,
            beam=None,
            lengths_key=None,
            inputs=None,
            version=None,
            return_labels=None,
    ):
        """A remote model with REST transport

        :param remote: The remote endpoint
        :param name:  The name of the model
        :param signature: The model signature
        :param labels: The labels (defaults to None)
        :param beam: The beam width (defaults to None)
        :param lengths_key: Which key is used for the length of the input vector (defaults to None)
        :param inputs: The inputs (defaults to empty list)
        :param version: The model version (defaults to None)
        :param return_labels: Whether the remote model returns class indices or the class labels directly. This depends
        on the `return_labels` parameter in exporters
        """
        super(RemoteModelREST, self).__init__(
            remote, name, signature, labels, beam, lengths_key, inputs, version, return_labels
        )
        url = urlparse(self.remote)
        if len(url.netloc.split(":")) != 2:
            raise ValueError("remote has to have the form <host_name>:<port>")
        self.hostname, self.port = url.netloc.split(":")
        v_str = '/versions/{}'.format(self.version) if self.version is not None else ''
        self.path = '/v1/models/{}{}:predict'.format(self.name, v_str)
        self.headers = {'Content-type': 'application/json'}

    def predict(self, examples, **kwargs):
        """Run prediction over HTTP/REST.

        :param examples: The input examples
        :return: The outcomes
        """

        verify_example(examples, self.input_keys)

        request = self.create_request(examples)

        conn = HTTPConnection(self.hostname, self.port)
        conn.request('POST', self.path, json.dumps(request), self.headers)
        response = conn.getresponse().read()
        outcomes_list = json.loads(response)
        if "error" in outcomes_list:
            raise ValueError("remote server returns error: {0}".format(outcomes_list["error"]))
        outcomes_list = outcomes_list["outputs"]
        outcomes_list = self.deserialize_response(examples, outcomes_list)
        return outcomes_list

    def deserialize_response(self, examples, predict_response):
        """Read the JSON response and decode it according to the signature.

        :param examples: Input examples
        :param predict_response: an HTTP/REST output
        """
        if self.signature == 'suggest_text':
            # s2s returns int values.
            tensor = np.array(predict_response)
            tensor = tensor.transpose(1, 2, 0)
            return tensor

        num_ex = len(predict_response['classes'])

        if self.signature == 'tag_text':
            classes = predict_response['classes']
            lengths = examples[self.lengths_key]
            result = []
            for i in range(num_ex):
                length_i = lengths[i]
                classes_i = classes[i]
                if self.return_labels:
                    d = [np.array(classes_i[j]) for j in range(length_i)]
                else:
                    d = [np.array(np.int32(classes_i[j])) for j in range(length_i)]
                result.append(d)

            return result

        if self.signature == 'predict_text':
            scores = predict_response['scores']
            classes = predict_response['classes']
            result = []
            for i in range(num_ex):
                score_i = scores[i]
                classes_i = classes[i]
                if self.return_labels:
                    d = [(c, np.float32(s)) for c, s in zip(classes_i, score_i)]
                else:
                    d = [(np.int32(c), np.float32(s)) for c, s in zip(classes_i, score_i)]
                result.append(d)
        return result


@exporter
class RemoteModelGRPC(RemoteModel):

    def __init__(self, remote, name, signature, labels=None, beam=None, lengths_key=None, inputs=None, version=None, return_labels=False):
        """A remote model with gRPC transport

        When using this type of model, there is an external dependency on the `grpc` package, as well as the
        TF serving protobuf stub files.  There is also currently a dependency on `tensorflow`

        :param remote: The remote endpoint
        :param name:  The name of the model
        :param signature: The model signature
        :param labels: The labels (defaults to None)
        :param beam: The beam width (defaults to None)
        :param lengths_key: Which key is used for the length of the input vector (defaults to None)
        :param inputs: The inputs (defaults to empty list)
        :param version: The model version (defaults to None)
        :param return_labels: Whether the remote model returns class indices or the class labels directly. This depends
        on the `return_labels` parameter in exporters
        """
        super(RemoteModelGRPC, self).__init__(
            remote, name, signature, labels, beam, lengths_key, inputs, version, return_labels
        )
        self.predictpb = import_user_module('tensorflow_serving.apis.predict_pb2')
        self.servicepb = import_user_module('tensorflow_serving.apis.prediction_service_pb2_grpc')
        self.metadatapb = import_user_module('tensorflow_serving.apis.get_model_metadata_pb2')
        self.grpc = import_user_module('grpc')
        self.channel = self.grpc.insecure_channel(remote)

    def decode_output(self, x):
        return x.decode('ascii') if self.return_labels else np.int32(x)

    def get_labels(self):
        """Return the model's labels

        :return: The model's labels
        """
        return self.labels

    def predict(self, examples, **kwargs):
        """Run prediction over gRPC

        :param examples: The input examples
        :return: The outcomes
        """
        verify_example(examples, self.input_keys)

        request = self.create_request(examples)
        stub = self.servicepb.PredictionServiceStub(self.channel)
        outcomes_list = stub.Predict(request)
        outcomes_list = self.deserialize_response(examples, outcomes_list)

        return outcomes_list

    def create_request(self, examples):
        # TODO: Remove TF dependency client side
        import tensorflow as tf

        request = self.predictpb.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = self.signature
        if self.version is not None:
            request.model_spec.version.value = self.version

        for feature in self.input_keys:
            if isinstance(examples[feature], np.ndarray):
                shape = examples[feature].shape
            else:
                shape = [1]

            dtype = examples[feature].dtype.type
            if issubclass(dtype, np.integer): dtype = tf.int32
            elif issubclass(dtype, np.floating): dtype = tf.float32
            else: dtype = tf.string

            tensor_proto = tf.contrib.util.make_tensor_proto(examples[feature], shape=shape, dtype=dtype)
            request.inputs[feature].CopyFrom(
                tensor_proto
            )

        return request

    def deserialize_response(self, examples, predict_response):
        """
        read the protobuf response from tensorflow serving and decode it according
        to the signature.

        here's the relevant piece of the proto:
            map<string, TensorProto> inputs = 2;

        the predict endpoint happens to have the ability to filter output for certain keys, but
        we do not support this currently. There are two keys we want to extract: classes and scores.

        :params predict_response: a PredictResponse protobuf object,
                    as defined in tensorflow_serving proto files
        """

        example_len = predict_response.outputs.get('classes').tensor_shape.dim[1].size
        num_examples = predict_response.outputs.get('classes').tensor_shape.dim[0].size

        if self.signature == 'suggest_text':
            # s2s returns int values.
            classes = predict_response.outputs.get('classes')
            shape = [dim.size for dim in classes.tensor_shape.dim]
            results = np.reshape(np.array(classes.int_val), shape)
            results = results.transpose(1, 2, 0)
            return results

        if self.signature == 'tag_text':
            if self.return_labels:
                classes = predict_response.outputs.get('classes').string_val
            else:
                classes = predict_response.outputs.get('classes').int_val
            lengths = examples[self.lengths_key]
            result = []
            for i in range(num_examples):
                length = lengths[i]
                tmp = [self.decode_output(x) for x in classes[example_len*i:example_len*(i+1)][:length]]
                result.append(tmp)

            return result

        if self.signature == 'predict_text':
            scores = predict_response.outputs.get('scores').float_val
            if self.return_labels:
                classes = predict_response.outputs.get('classes').string_val
            else:
                classes = predict_response.outputs.get('classes').int_val
            result = []
            length = len(self.get_labels())
            for i in range(num_examples):   # wrap in numpy because the local models send that dtype out
                d = [(self.decode_output(c), np.float32(s)) for c, s in zip(classes[example_len*i:example_len*(i+1)][:length], scores[length*i:length*(i+1)][:length])]
                result.append(d)
            return result
