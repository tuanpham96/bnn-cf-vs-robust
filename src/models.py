import tensorflow as tf

from tensorflow.keras import layers, initializers
from tensorflow.keras import Model, Sequential

@tf.custom_gradient
def SignActivation(x):
    y = tf.sign(x)
    def grad(grad_output):
        # use hardtanh gradient (see paper): anything with abs() > 1 -> 0
        return tf.where(tf.abs(x) > 1, 0, grad_output)
    return y, grad

def Binarize(tensor):
    return tf.sign(tensor)

class BinarizeLinear(layers.Layer):
    '''
    units:          number of units for layer
    init_type:      'gauss' or 'uniform' for weight initialization in Dense layer
    init_width:     used for initialization widths or stddev for weight initialization
    dropout_rate:   if not None, will use to construct Dropout layer
    act_fun:        if not None, activation function; currently only 'sign' (SignActivation)
    norm_type:      if not None, will use for normalization layer; currently only 'bn' (batchnorm)
    bin_inp:        (TODO) whether to normalize input; in paper says NO but unclear in code does
    '''
    def __init__(self, units, init_type = 'gauss', init_width = 0.01,
                 dropout_rate = None, norm_type = 'bn', act_fun = 'sign',
                 bin_inp = False, name = 'bfc'):

        super(BinarizeLinear, self).__init__()
        self.units = units
        self.init_type = init_type
        self.init_width = init_width
        self.dropout_rate = dropout_rate
        self.act_fun = act_fun
        self.norm_type = norm_type
        self.bin_inp = bin_inp

    def get_dense_initializer(self):
        kernel_init = 'glorot_uniform'
        init_width = self.init_width
        if self.init_type == 'gauss':
            kernel_init = initializers.RandomNormal(mean=0.0, stddev=init_width)
        if self.init_type == 'uniform':
            kernel_init = initializers.RandomUniform(minval=-init_width/2, maxval=init_width/2)
        return kernel_init

    def build(self, input_shape):
        self.inp_dim = input_shape

        # Dense linear layer
        self.fc = layers.Dense(self.units, use_bias=False, activation=None,
                                kernel_initializer=self.get_dense_initializer())
        self.fc.build(input_shape)

        # Create 'org' (i.e. hidden weight) in weight and binarize
        if not hasattr(self.fc.kernel,'org'):
            self.fc.kernel.org = tf.identity(self.fc.kernel)
        self.fc.kernel.assign(Binarize(self.fc.kernel.org))

        # Create dropout layer
        if self.dropout_rate:
            self.dropout = layers.Dropout(rate=self.dropout_rate)

        # Create normalization layer
        if self.norm_type:
            if self.norm_type == 'bn':
                self.norm = layers.BatchNormalization()
            else:
                raise NotImplementedError

        # Acitvation
        if self.act_fun:
            if self.act_fun == 'sign':
                self.act = SignActivation
            else:
                raise NotImplementedError


    def call(self, input):
        # TODO: unclear why in paper says no but here performs binarize
        # plus this is a tad hardcoded
        if input.shape[1] != 784 and self.bin_inp:
            input = Binarize(input)

        self.fc.kernel.assign(Binarize(self.fc.kernel.org))
        out = self.fc(input)

        if self.dropout_rate:
            out = self.dropout(out)

        if self.norm_type:
            out = self.norm(out)

        if self.act_fun:
            out = self.act(out)

        return out

class BNN(Model):
    '''
    layers_dims:    [(input_height, input_width), hidden_1, hidden_2, ..., output]
    '''
    def __init__(self, layers_dims, **kwargs):
        super(BNN, self).__init__()
        self.layers_dims = layers_dims
        self.num_hidden = len(layers_dims) - 2

        self.hidden_args = dict(**kwargs)

        self.output_args = dict(**kwargs)
        self.output_args['act_fun'] = None # no activation at output

        # define layers
        self.flatten = layers.Flatten(input_shape=layers_dims[0])
        self.bfcs = Sequential([
            BinarizeLinear(layers_dims[i],
                           **self.hidden_args,
                           name = 'bfc-%02d' %(i))
            for i in range(1, self.num_hidden+1)
        ])
        self.out = BinarizeLinear(layers_dims[-1], **self.output_args, name='output')

    def call(self, x):
        x = self.flatten(x)
        x = self.bfcs(x)
        x = self.out(x)
        return x

