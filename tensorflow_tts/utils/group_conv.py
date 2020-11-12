# -*- coding: utf-8 -*-
# This code is copy from https://github.com/tensorflow/tensorflow/pull/36773.
"""Group Convolution Modules."""

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import Conv1D, SeparableConv1D
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn, nn_ops


class Convolution(object):
    """Helper class for convolution.
    Note that this class assumes that shapes of input and filter passed to
    __call__ are compatible with input_shape and filter_shape passed to the
    constructor.
    Arguments
      input_shape: static shape of input. i.e. input.get_shape().
      filter_shape: static shape of the filter. i.e. filter.get_shape().
      padding:  see convolution.
      strides: see convolution.
      dilation_rate: see convolution.
      name: see convolution.
      data_format: see convolution.
    """

    def __init__(
        self,
        input_shape,
        filter_shape,
        padding,
        strides=None,
        dilation_rate=None,
        name=None,
        data_format=None,
    ):
        """Helper function for convolution."""
        num_total_dims = filter_shape.ndims
        if num_total_dims is None:
            num_total_dims = input_shape.ndims
        if num_total_dims is None:
            raise ValueError("rank of input or filter must be known")

        num_spatial_dims = num_total_dims - 2

        try:
            input_shape.with_rank(num_spatial_dims + 2)
        except ValueError:
            raise ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

        try:
            filter_shape.with_rank(num_spatial_dims + 2)
        except ValueError:
            raise ValueError("filter tensor must have rank %d" % (num_spatial_dims + 2))

        if data_format is None or not data_format.startswith("NC"):
            input_channels_dim = tensor_shape.dimension_at_index(
                input_shape, num_spatial_dims + 1
            )
            spatial_dims = range(1, num_spatial_dims + 1)
        else:
            input_channels_dim = tensor_shape.dimension_at_index(input_shape, 1)
            spatial_dims = range(2, num_spatial_dims + 2)

        filter_dim = tensor_shape.dimension_at_index(filter_shape, num_spatial_dims)
        if not (input_channels_dim % filter_dim).is_compatible_with(0):
            raise ValueError(
                "number of input channels is not divisible by corresponding "
                "dimension of filter, {} % {} != 0".format(
                    input_channels_dim, filter_dim
                )
            )

        strides, dilation_rate = nn_ops._get_strides_and_dilation_rate(
            num_spatial_dims, strides, dilation_rate
        )

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.data_format = data_format
        self.strides = strides
        self.padding = padding
        self.name = name
        self.dilation_rate = dilation_rate
        self.conv_op = nn_ops._WithSpaceToBatch(
            input_shape,
            dilation_rate=dilation_rate,
            padding=padding,
            build_op=self._build_op,
            filter_shape=filter_shape,
            spatial_dims=spatial_dims,
            data_format=data_format,
        )

    def _build_op(self, _, padding):
        return nn_ops._NonAtrousConvolution(
            self.input_shape,
            filter_shape=self.filter_shape,
            padding=padding,
            data_format=self.data_format,
            strides=self.strides,
            name=self.name,
        )

    def __call__(self, inp, filter):
        return self.conv_op(inp, filter)


class Conv(Layer):
    """Abstract N-D convolution layer (private, used as implementation base).
    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    Arguments:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      groups: Integer, the number of channel groups controlling the connections
        between inputs and outputs. Input channels and `filters` must both be
        divisible by `groups`. For example,
          - At `groups=1`, all inputs are convolved to all outputs.
          - At `groups=2`, the operation becomes equivalent to having two
            convolutional layers side by side, each seeing half the input
            channels, and producing half the output channels, and both
            subsequently concatenated.
          - At `groups=input_channels`, each input channel is convolved with its
            own set of filters, of size `input_channels / filters`
      activation: Activation function to use.
        If you don't specify anything, no activation is applied.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs
    ):
        super(Conv, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs
        )
        self.rank = rank
        if filters is not None and not isinstance(filters, int):
            filters = int(filters)
        self.filters = filters
        self.groups = groups or 1
        if filters is not None and filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the number of "
                "groups. Received: groups={}, filters={}".format(groups, filters)
            )
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0(s). "
                "Received: %s" % (kernel_size,)
            )
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding == "causal" and not isinstance(self, (Conv1D, SeparableConv1D)):
            raise ValueError(
                "Causal padding is only supported for `Conv1D`"
                "and ``SeparableConv1D`."
            )
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by the number "
                "of groups. Received groups={}, but the input has {} channels "
                "(full input shape is {}).".format(
                    self.groups, input_channel, input_shape
                )
            )
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_channel}
        )

        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2
        )
        self._convolution_op = Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self._padding_op,
            data_format=self._conv_op_data_format,
        )
        self.built = True

    def call(self, inputs):
        if self._recreate_conv_op(inputs):
            self._convolution_op = Convolution(
                inputs.get_shape(),
                filter_shape=self.kernel.shape,
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self._padding_op,
                data_format=self._conv_op_data_format,
            )
            self._build_conv_op_input_shape = inputs.get_shape()

        # Apply causal padding to inputs for Conv1D.
        if self.padding == "causal" and self.__class__.__name__ == "Conv1D":
            inputs = array_ops.pad(inputs, self._compute_causal_padding())

        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            if self.data_format == "channels_first":
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format="NCHW")
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format="NHWC")

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tensor_shape.TensorShape(
                [input_shape[0]] + new_space + [self.filters]
            )
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] + new_space)

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == "channels_last":
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == "causal":
            op_padding = "valid"
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding

    def _recreate_conv_op(self, inputs):
        """Recreate conv_op if necessary.
        Check if the input_shape in call() is different from that in build().
        For the values that are not None, if they are different, recreate
        the _convolution_op to avoid the stateful behavior.
        Args:
          inputs: The input data to call() method.
        Returns:
          `True` or `False` to indicate whether to recreate the conv_op.
        """
        call_input_shape = inputs.get_shape()
        for axis in range(1, len(call_input_shape)):
            if (
                call_input_shape[axis] is not None
                and self._build_conv_op_input_shape[axis] is not None
                and call_input_shape[axis] != self._build_conv_op_input_shape[axis]
            ):
                return True
        return False


class GroupConv1D(Conv):
    """1D convolution layer (e.g. temporal convolution).
    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.
    Examples:
    >>> # The inputs are 128-length vectors with 10 timesteps, and the batch size
    >>> # is 4.
    >>> input_shape = (4, 10, 128)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv1D(
    ... 32, 3, activation='relu',input_shape=input_shape)(x)
    >>> print(y.shape)
    (4, 8, 32)
    Arguments:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
        specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
        `"causal"` results in causal (dilated) convolutions, e.g. `output[t]`
        does not depend on `input[t+1:]`. Useful when modeling temporal data
        where the model should not violate the temporal order.
        See [WaveNet: A Generative Model for Raw Audio, section
          2.1](https://arxiv.org/abs/1609.03499).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
      groups: Integer, the number of channel groups controlling the connections
        between inputs and outputs. Input channels and `filters` must both be
        divisible by `groups`. For example,
          - At `groups=1`, all inputs are convolved to all outputs.
          - At `groups=2`, the operation becomes equivalent to having two
            convolutional layers side by side, each seeing half the input
            channels, and producing half the output channels, and both
            subsequently concatenated.
          - At `groups=input_channels`, each input channel is convolved with its
            own set of filters, of size `input_channels / filters`
      dilation_rate: an integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied (
        see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix (
        see `keras.initializers`).
      bias_initializer: Initializer for the bias vector (
        see `keras.initializers`).
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (
        see `keras.regularizers`).
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation") (
        see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix (
        see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (
        see `keras.constraints`).
    Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    Returns:
      A tensor of rank 3 representing
      `activation(conv1d(inputs, kernel) + bias)`.
    Raises:
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs
        )
