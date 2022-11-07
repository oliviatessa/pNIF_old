__all__ = ["NIFMultiScale", "NIF", "NIFMultiScaleLastLayerParameterized", "PNIF"]

import tensorflow as tf
from tensorflow.keras import Model, initializers
from .layers import *
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

class NIF(Model):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy='float32'):
        super(NIF, self).__init__()
        self.cfg_shape_net = cfg_shape_net
        self.si_dim = cfg_shape_net['input_dim']
        self.so_dim = cfg_shape_net['output_dim']
        self.n_sx = cfg_shape_net['units']
        self.l_sx = cfg_shape_net['nlayers']
        self.pi_dim = cfg_parameter_net['input_dim']
        self.pi_hidden = cfg_parameter_net['latent_dim']
        self.n_st = cfg_parameter_net['units']
        self.l_st = cfg_parameter_net['nlayers']

        self.mixed_policy = tf.keras.mixed_precision.experimental.Policy(mixed_policy) # policy object can be feed into keras.layer
        self.variable_Dtype = self.mixed_policy.variable_dtype
        self.compute_Dtype = self.mixed_policy.compute_dtype

        # initialize the parameter net structure
        self.pnet_list = self._initialize_pnet(cfg_parameter_net, cfg_shape_net)


    def call(self, inputs, training=None, mask=None):
        input_p = inputs[:, 0:self.pi_dim]
        input_s = inputs[:, self.pi_dim:self.pi_dim+self.si_dim]
        # get parameter from parameter_net
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net(tf.cast(input_s,self.compute_Dtype),
                                    self.pnet_output,
                                    si_dim=self.si_dim,
                                    so_dim=self.so_dim,
                                    n_sx=self.n_sx,
                                    l_sx=self.l_sx,
                                    activation=self.cfg_shape_net['activation'],
                                    variable_dtype=self.variable_Dtype)

    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def _initialize_pnet(self, cfg_parameter_net, cfg_shape_net):
        # just simple implementation of a shortcut connected parameter net with a similar shapenet
        self.po_dim = (self.l_sx)*self.n_sx**2 + (self.si_dim + self.so_dim + 1 + self.l_sx)*self.n_sx + self.so_dim

        # construct parameter_net
        pnet_layers_list = []
        # 1. first layer
        layer_1 = Dense(self.n_st, cfg_parameter_net['activation'],
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                        dtype=self.mixed_policy)
        pnet_layers_list.append(layer_1)

        # 2. hidden layer
        for i in range(self.l_st):
            tmp_layer = MLP_SimpleShortCut(self.n_st, cfg_parameter_net['activation'],
                                           kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                           bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                                           mixed_policy=self.mixed_policy)
            # identity_layer = Lambda(lambda x: x)
            # tmp_layer =tf.keras.layers.Add()(identity_layer,tmp_layer)
            pnet_layers_list.append(tmp_layer)

        # 3. bottleneck layer
        bottleneck_layer = Dense(self.pi_hidden,
                                 kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                 bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                                 dtype=self.mixed_policy)
        pnet_layers_list.append(bottleneck_layer)

        # 4. last layer
        last_layer = Dense(self.po_dim,
                           kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                           dtype=self.mixed_policy)
        pnet_layers_list.append(last_layer)

        return pnet_layers_list

    @staticmethod
    def _call_shape_net(input_s, pnet_output, si_dim, so_dim, n_sx, l_sx, activation, variable_dtype):
        w_1 = tf.reshape(pnet_output[:, :si_dim*n_sx],
                         [-1, si_dim, n_sx])
        w_hidden_list = []
        for i in range(l_sx):
            w_tmp = tf.reshape(pnet_output[:,
                               si_dim*n_sx + i*n_sx**2:
                               si_dim*n_sx + (i + 1)*n_sx**2],
                               [-1, n_sx, n_sx])
            w_hidden_list.append(w_tmp)
        w_l = tf.reshape(pnet_output[:,
                         si_dim*n_sx + l_sx*n_sx**2:
                         si_dim*n_sx + l_sx*n_sx**2 + so_dim*n_sx],
                         [-1, n_sx, so_dim])
        n_weights = si_dim*n_sx + l_sx*n_sx**2 + so_dim*n_sx

        # distribute bias
        b_1 = tf.reshape(pnet_output[:, n_weights: n_weights + n_sx],
                         [-1, n_sx])
        b_hidden_list = []
        for i in range(l_sx):
            b_tmp = tf.reshape(pnet_output[:, n_weights + n_sx + i*n_sx:
                                              n_weights + n_sx + (i + 1)*n_sx], [-1, n_sx])
            b_hidden_list.append(b_tmp)
        b_l = tf.reshape(pnet_output[:,
                         n_weights + (l_sx + 1)*n_sx:],
                         [-1, so_dim])

        # construct shape net
        act_fun = tf.keras.activations.get(activation)
        u = act_fun(tf.einsum('ai,aij->aj', input_s, w_1) + b_1)

        for i in range(l_sx):
            w_tmp = w_hidden_list[i]
            b_tmp = b_hidden_list[i]
            u = act_fun(tf.einsum('ai,aij->aj', u, w_tmp) + b_tmp) + u
        u = tf.einsum('ai,aij->aj', u, w_l) + b_l
        return tf.cast(u, variable_dtype)

    @staticmethod
    def _call_parameter_net(input_p, pnet_list):
        latent = input_p
        for l in pnet_list[:-1]:
            latent = l(latent)
        output_final = pnet_list[-1](latent)
        return output_final, latent

    def model(self):
        input_tot = tf.keras.layers.Input(shape=(self.si_dim + self.pi_dim), name='input')
        return Model(inputs=[input_tot], outputs=[self.call(input_tot)])

    def model_p_to_lr(self):
        input_p = tf.keras.layers.Input(shape=(self.pi_dim))
        # this model: t, mu -> hidden LR
        return Model(inputs=[input_p], outputs=[self._call_parameter_net(input_p, self.pnet_list)[1]])

    def model_lr_to_w(self):
        input_lr = tf.keras.layers.Input(shape=(self.pi_hidden))
        # this model: hidden LR -> weights and biases of shapenet
        return Model(inputs=[input_lr],outputs=[self.pnet_list[-1](input_lr)])

    def model_x_to_u_given_w(self):
        input_s = tf.keras.layers.Input(shape=(self.si_dim), dtype=self.variable_Dtype)
        input_pnet = tf.keras.layers.Input(shape=(self.pnet_list[-1].output_shape[1]), dtype=self.variable_Dtype)
        return Model(inputs=[input_s, input_pnet],
                     outputs=[self._call_shape_net(tf.cast(input_s,self.compute_Dtype),
                                                   tf.cast(input_pnet,self.compute_Dtype),
                                                   si_dim=self.si_dim,
                                                   so_dim=self.so_dim,
                                                   n_sx=self.n_sx,
                                                   l_sx=self.l_sx,
                                                   activation=self.cfg_shape_net['activation'],
                                                   variable_dtype=self.variable_Dtype)])

class PNIF(NIF):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy):
        super(PNIF, self).__init__(cfg_shape_net, cfg_parameter_net, mixed_policy)
        
    def call(self, inputs, training=None, mask=None):
        input_p = inputs[:, 0:self.pi_dim]
        input_s = inputs[:, self.pi_dim:self.pi_dim+self.si_dim]
        # get parameter from parameter_net
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net(tf.cast(input_s,self.compute_Dtype),
                                    self.pnet_output,
                                    si_dim=self.si_dim,
                                    so_dim=self.so_dim,
                                    n_sx=self.n_sx,
                                    l_sx=self.l_sx,
                                    activation=self.cfg_shape_net['activation'],
                                    variable_dtype=self.variable_Dtype)

    def _initialize_pnet(self, cfg_parameter_net, cfg_shape_net):
        # just simple implementation of a shortcut connected parameter net with a similar shapenet
        self.po_dim = (self.l_sx)*self.n_sx**2 + (self.si_dim + self.so_dim + 1 + self.l_sx)*self.n_sx + self.so_dim

        # construct parameter_net
        pnet_layers_list = []
        # 1. first layer
        layer_1 = MaskLayer(self.pi_dim, self.n_st, cfg_parameter_net['activation'],
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        bias_initializer=initializers.TruncatedNormal(stddev=0.1))
        pnet_layers_list.append(layer_1)

        # 2. hidden layer
        for i in range(self.l_st):
            tmp_layer = MaskLayer(self.n_st, self.n_st, cfg_parameter_net['activation'],
                                           kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                           bias_initializer=initializers.TruncatedNormal(stddev=0.1))
            # identity_layer = Lambda(lambda x: x)
            # tmp_layer =tf.keras.layers.Add()(identity_layer,tmp_layer)
            pnet_layers_list.append(tmp_layer)

        # 3. bottleneck layer
        bottleneck_layer = MaskLayer(self.n_st, self.pi_hidden, activation=None,
                                 kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                 bias_initializer=initializers.TruncatedNormal(stddev=0.1))
        pnet_layers_list.append(bottleneck_layer)

        # 4. last layer
        last_layer = MaskLayer(self.pi_hidden, self.po_dim, activation=None,
                           kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=initializers.TruncatedNormal(stddev=0.1))
        pnet_layers_list.append(last_layer)

        return pnet_layers_list
    
    def update_masks(self, sparsity):
        '''
        Add keyword like "Magnitude" to choose which type of pruning you want to do 
        '''
        for layer in self.pnet_list[:-1]:
            #omit last layer because it is not prunable 
            layer.pruneLowMagnitude(sparsity)
          
        
        self.pnet_list[-1].pruneShapeNet(sparsity, self.si_dim, self.n_sx, self.l_sx, self.so_dim)
            
        '''
        Make loop that does model.fit for however many times we want to prune
        call ori_model.update_masks each time 
        '''


class NIFMultiScale(NIF):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy='float32'):
        super(NIFMultiScale, self).__init__(cfg_shape_net, cfg_parameter_net, mixed_policy)

    def call(self, inputs, training=None, mask=None):
            input_p = inputs[:, 0:self.pi_dim]
            input_s = inputs[:, self.pi_dim:self.pi_dim+self.si_dim]
            # get parameter from parameter_net
            self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
            return self._call_shape_net_mres(tf.cast(input_s, self.compute_Dtype),
                                             self.pnet_output,
                                             flag_resblock=self.cfg_shape_net['use_resblock'],
                                             omega_0=tf.cast(self.cfg_shape_net['omega_0'], self.compute_Dtype),
                                             si_dim=self.si_dim,
                                             so_dim=self.so_dim,
                                             n_sx=self.n_sx,
                                             l_sx=self.l_sx,
                                             variable_dtype=self.variable_dtype
                                             )

    def _initialize_pnet(self, cfg_parameter_net, cfg_shape_net):
        """
        generate the layers for parameter net, given configuration of
        shape_net you will also need the last layer to be consistent with
        the total number of shapenet' weights+biases
        """

        if not isinstance(cfg_parameter_net, dict):
            raise TypeError("cfg_parameter_net must be a dictionary")
        if not isinstance(cfg_shape_net, dict):
            raise TypeError("cfg_shape_net must be a dictionary")
        assert 'use_resblock' in cfg_shape_net.keys(), "`use_resblock` should be in cfg_shape_net"
        # assert 'nn_type' in cfg_parameter_net.keys(), "`nn_type` should be in cfg_parameter_net"
        assert type(cfg_shape_net['use_resblock']) == bool, "cfg_shape_net['use_resblock'] must be a bool"

        pnet_layers_list = []
        if cfg_shape_net['connectivity'] == 'full':
            # very first, determine the output dimension of parameter_net
            if cfg_shape_net['use_resblock']:
                self.po_dim = (2*self.l_sx)*self.n_sx**2 + (self.si_dim + self.so_dim + 1 + 2*self.l_sx)*self.n_sx + self.so_dim
            else:
                self.po_dim = (self.l_sx)*self.n_sx**2 + (self.si_dim + self.so_dim + 1 + self.l_sx)*self.n_sx + self.so_dim
        elif cfg_shape_net['connectivity'] == 'last_layer':
            # only parameterize the last layer
            self.po_dim = self.pi_hidden
        else:
            raise ValueError("cfg_shape_net missing correct `connectivity`")

        # first, and hidden layers are only dependent on the type of parameter_net
        # if cfg_parameter_net['nn_type'] == 'siren':
        if cfg_parameter_net['activation'] == 'sine':
            # assert cfg_parameter_net['activation'] == 'sine', "you should specify activation in cfg_parameter_net as " \
            #                                                   "sine"
            # 1. first layer
            layer_1 = SIREN(self.pi_dim, self.n_st, 'first',
                            cfg_parameter_net['omega_0'],
                            cfg_shape_net,
                            self.mixed_policy)
            pnet_layers_list.append(layer_1)

            # 2. hidden layers
            if cfg_parameter_net['use_resblock']:
                for i in range(self.l_st):
                    tmp_layer = SIREN_ResNet(self.n_st, self.n_st,
                                             cfg_parameter_net['omega_0'],
                                             self.mixed_policy)
                    pnet_layers_list.append(tmp_layer)
            else:
                for i in range(self.l_sx):
                    tmp_layer = SIREN(self.n_st, self.n_st, 'hidden',
                                      cfg_parameter_net['omega_0'],
                                      cfg_shape_net,
                                      self.mixed_policy)
                    pnet_layers_list.append(tmp_layer)

            # 3. bottleneck layer
            bottleneck_layer = SIREN(self.n_st, self.pi_hidden, 'bottleneck',
                                     cfg_parameter_net['omega_0'],
                                     cfg_shape_net,
                                     self.mixed_policy)
            pnet_layers_list.append(bottleneck_layer)

            # 4. last layer
            last_layer = HyperLinearForSIREN(self.pi_hidden, self.po_dim,
                                             cfg_shape_net,
                                             self.mixed_policy,
                                             connectivity=cfg_shape_net['connectivity'])
            # last_layer = SIREN(self.pi_hidden, self.po_dim, 'last',
            #                    cfg_parameter_net['omega_0'],
            #                    cfg_shape_net['omega_0'], cfg_shape_net,
            #                    self.mixed_policy)

            pnet_layers_list.append(last_layer)

        else:
            # cfg_parameter_net['nn_type'] == 'mlp':
            # 1. first layer
            layer_1 = Dense(self.n_st, cfg_parameter_net['activation'],
                            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                            dtype=self.mixed_policy)
            pnet_layers_list.append(layer_1)

            # 2. hidden layer
            if cfg_parameter_net['use_resblock']:
                for i in range(self.l_st):
                    tmp_layer = MLP_ResNet(width=self.n_st,
                                           activation=cfg_parameter_net['activation'],
                                           kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                           bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                                           mixed_policy=self.mixed_policy)
                    pnet_layers_list.append(tmp_layer)
            else:
                for i in range(self.l_st):
                    tmp_layer = MLP_SimpleShortCut(self.n_st, cfg_parameter_net['activation'],
                                                   kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                                   bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                                                   mixed_policy=self.mixed_policy)
                    # identity_layer = Lambda(lambda x: x)
                    # tmp_layer =tf.keras.layers.Add()(identity_layer,tmp_layer)
                    pnet_layers_list.append(tmp_layer)

            # 3. bottleneck layer
            bottleneck_layer = Dense(self.pi_hidden,
                                     kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                                     bias_initializer=initializers.TruncatedNormal(stddev=0.1),
                                     dtype=self.mixed_policy)
            pnet_layers_list.append(bottleneck_layer)

            # 4. last layer
            last_layer = HyperLinearForSIREN(self.pi_hidden, self.po_dim,
                                             cfg_shape_net,
                                             self.mixed_policy,
                                             connectivity=cfg_shape_net['connectivity'])
            pnet_layers_list.append(last_layer)

        return pnet_layers_list

    @staticmethod
    def _call_shape_net_mres(input_s, pnet_output, flag_resblock, omega_0, si_dim, so_dim, n_sx, l_sx, variable_dtype):
        """
        distribute `pnet_output` into weight and bias, it depends on the type of shapenet.

        For now, we only support shapenet having the following structure,
            - resnet-block
            - plain fnn
        """
        if flag_resblock:
            # distribute weights
            w_1 = tf.reshape(pnet_output[:, :si_dim*n_sx],
                             [-1, si_dim, n_sx])
            w_hidden_list = []
            for i in range(l_sx):
                w1_tmp = tf.reshape(pnet_output[:,
                                    si_dim*n_sx + 2*i*n_sx**2:
                                    si_dim*n_sx + (2*i + 1)*n_sx**2],
                                    [-1, n_sx, n_sx])
                w2_tmp = tf.reshape(pnet_output[:,
                                    si_dim*n_sx + (2*i + 1)*n_sx**2:
                                    si_dim*n_sx + (2*i + 2)*n_sx**2],
                                    [-1, n_sx, n_sx])
                w_hidden_list.append([w1_tmp, w2_tmp])
            w_l = tf.reshape(pnet_output[:,
                             si_dim*n_sx + (2*l_sx)*n_sx**2:
                             si_dim*n_sx + (2*l_sx)*n_sx**2 + so_dim*n_sx],
                             [-1, n_sx, so_dim])

            n_weights = si_dim*n_sx + (2*l_sx)*n_sx**2 + so_dim*n_sx

            # distribute bias
            b_1 = tf.reshape(pnet_output[:, n_weights: n_weights + n_sx],
                             [-1, n_sx])
            b_hidden_list = []
            for i in range(l_sx):
                b1_tmp = tf.reshape(pnet_output[:,
                                    n_weights + n_sx + 2*i*n_sx:
                                    n_weights + n_sx + (2*i + 1)*n_sx],
                                    [-1, n_sx])
                b2_tmp = tf.reshape(pnet_output[:,
                                    n_weights + n_sx + (2*i + 1)*n_sx:
                                    n_weights + n_sx + (2*i + 2)*n_sx],
                                    [-1, n_sx])
                b_hidden_list.append([b1_tmp, b2_tmp])
            b_l = tf.reshape(pnet_output[:, n_weights + (2*l_sx + 1)*n_sx:], [-1, so_dim])

            # construct shape net
            u = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', input_s, w_1) + b_1)
            for i in range(l_sx):
                h = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', u, w_hidden_list[i][0]) + b_hidden_list[i][0])
                u = 0.5*(u + tf.math.sin(omega_0*tf.einsum('ai,aij->aj', h, w_hidden_list[i][1]) + b_hidden_list[i][1]))
            u = tf.einsum('ai,aij->aj', u, w_l) + b_l

        else:
            # distribute weights
            w_1 = tf.reshape(pnet_output[:, :si_dim*n_sx],
                             [-1, si_dim, n_sx])
            w_hidden_list = []
            for i in range(l_sx):
                w_tmp = tf.reshape(pnet_output[:,
                                   si_dim*n_sx + i*n_sx**2:
                                   si_dim*n_sx + (i + 1)*n_sx**2],
                                   [-1, n_sx, n_sx])
                w_hidden_list.append(w_tmp)
            w_l = tf.reshape(pnet_output[:,
                             si_dim*n_sx + l_sx*n_sx**2:
                             si_dim*n_sx + l_sx*n_sx**2 + so_dim*n_sx],
                             [-1, n_sx, so_dim])
            n_weights = si_dim*n_sx + l_sx*n_sx**2 + so_dim*n_sx

            # distribute bias
            b_1 = tf.reshape(pnet_output[:, n_weights: n_weights + n_sx],
                             [-1, n_sx])
            b_hidden_list = []
            for i in range(l_sx):
                b_tmp = tf.reshape(pnet_output[:, n_weights + n_sx + i*n_sx:
                                                  n_weights + n_sx + (i + 1)*n_sx], [-1, n_sx])
                b_hidden_list.append(b_tmp)
            b_l = tf.reshape(pnet_output[:,
                             n_weights + (l_sx + 1)*n_sx:],
                             [-1, so_dim])

            # construct shape net
            u = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', input_s, w_1) + b_1)
            for i in range(l_sx):
                u = tf.math.sin(omega_0*tf.einsum('ai,aij->aj', u, w_hidden_list[i]) + b_hidden_list[i])
            u = tf.einsum('ai,aij->aj', u, w_l) + b_l

        return tf.cast(u, variable_dtype)

    def model_x_to_u_given_w(self):
        input_s = tf.keras.layers.Input(shape=(self.si_dim))
        input_pnet = tf.keras.layers.Input(shape=(self.pnet_list[-1].output_shape[1]))
        return Model(inputs=[input_s, input_pnet],
                     outputs=[self._call_shape_net_mres(tf.cast(input_s, self.compute_Dtype),
                                                        tf.cast(input_pnet, self.compute_Dtype),
                                                        flag_resblock=self.cfg_shape_net['use_resblock'],
                                                        omega_0=tf.cast(self.cfg_shape_net['omega_0'],
                                                                        self.compute_Dtype),
                                                        si_dim=self.si_dim,
                                                        so_dim=self.so_dim,
                                                        n_sx=self.n_sx,
                                                        l_sx=self.l_sx,
                                                        variable_dtype=self.variable_dtype
                                                        )])


class NIFMultiScaleLastLayerParameterized(NIFMultiScale):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy='float32'):
        super(NIFMultiScaleLastLayerParameterized, self).__init__(cfg_shape_net, cfg_parameter_net, mixed_policy)
        assert cfg_shape_net['connectivity'] == 'last_layer'
        self.snet_list, self.last_layer_bias = self._initialize_snet(cfg_shape_net, cfg_parameter_net)

    def call(self, inputs, training=None, mask=None):
        input_p = inputs[:, 0:self.pi_dim]
        input_s = inputs[:, self.pi_dim:self.pi_dim+self.si_dim]
        # get parameter from parameter_net
        self.pnet_output = self._call_parameter_net(input_p, self.pnet_list)[0]
        return self._call_shape_net_mres_only_para_last_layer(tf.cast(input_s, self.compute_Dtype),
                                                              self.snet_list,
                                                              tf.cast(self.last_layer_bias, self.compute_Dtype),
                                                              self.pnet_output,
                                                              self.so_dim,
                                                              self.pi_hidden,
                                                              self.variable_dtype)

    def model_p_to_lr(self):
        input_p = tf.keras.layers.Input(shape=(self.pi_dim))
        # this model: t, mu -> hidden LR
        return Model(inputs=[input_p], outputs=[self._call_parameter_net(input_p, self.pnet_list)[0]])

    def model_x_to_phi(self):
        input_s = tf.keras.layers.Input(shape=(self.si_dim))
        return Model(inputs=[input_s],
                     outputs=[tf.cast(self._call_shape_net_get_phi_x(input_s, self.snet_list, self.so_dim,
                                                                     self.pi_hidden), self.variable_Dtype)])

    def model_lr_to_w(self):
        raise ValueError("In this class: NIFMultiScaleLastLayerParameterization, `w` is the same as `lr`")

    def model_x_to_u_given_w(self):
        input_s = tf.keras.layers.Input(shape=(self.si_dim))
        input_pnet = tf.keras.layers.Input(shape=(self.pnet_list[-1].output_shape[1]))
        return Model(inputs=[input_s, input_pnet],
                     outputs=[self._call_shape_net_mres_only_para_last_layer(tf.cast(input_s, self.compute_Dtype),
                                                                             self.snet_list,
                                                                             tf.cast(self.last_layer_bias, self.compute_Dtype),
                                                                             tf.cast(input_pnet, self.compute_Dtype),
                                                                             self.so_dim,
                                                                             self.pi_hidden,
                                                                             self.variable_dtype)])



    def _initialize_snet(self, cfg_shape_net, cfg_parameter_net):
        # create a simple feedfowrard, with resblock or not, that maps self.si_dim to
        # self.so_dim*self.pi_hidden

        snet_layers_list = []
        # 1. first layer
        layer_1 = SIREN(self.si_dim, self.n_sx, 'first',
                        cfg_shape_net['omega_0'],
                        cfg_shape_net,
                        self.mixed_policy)
        snet_layers_list.append(layer_1)

        # 2. hidden layers
        if cfg_shape_net['use_resblock']:
            for i in range(self.l_sx):
                tmp_layer = SIREN_ResNet(self.n_sx, self.n_sx,
                                         cfg_shape_net['omega_0'],
                                         self.mixed_policy)
                snet_layers_list.append(tmp_layer)
        else:
            for i in range(self.l_sx):
                tmp_layer = SIREN(self.n_sx, self.n_sx, 'hidden',
                                  cfg_shape_net['omega_0'],
                                  cfg_shape_net,
                                  self.mixed_policy)
                snet_layers_list.append(tmp_layer)

        # 3. bottleneck AND the same time, last layer for spatial basis
        bottle_last_layer = SIREN(self.n_sx, self.po_dim*self.so_dim, 'bottleneck',
                                  cfg_shape_net['omega_0'],
                                  cfg_shape_net,
                                  self.mixed_policy)
        snet_layers_list.append(bottle_last_layer)

        # create bias for the last layer
        last_layer_init = initializers.TruncatedNormal(stddev=0.1)
        last_layer_bias = tf.Variable(last_layer_init([self.so_dim,]), dtype=self.mixed_policy.variable_dtype)

        return snet_layers_list, last_layer_bias

    def _call_shape_net_get_phi_x(self, input_s, snet_layers_list, so_dim, pi_hidden):
        # 1. x -> phi_x
        phi_x = input_s
        for l in snet_layers_list:
            phi_x = l(phi_x)
        # 2. phi_x * a_t + bias
        phi_x_matrix = tf.reshape(phi_x, [-1, so_dim, pi_hidden])
        return phi_x_matrix

    def _call_shape_net_mres_only_para_last_layer(self, input_s, snet_layers_list, last_layer_bias, pnet_output,
                                                  so_dim, pi_hidden, variable_dtype):
        phi_x_matrix = self._call_shape_net_get_phi_x(input_s, snet_layers_list, so_dim, pi_hidden)
        u = tf.keras.layers.Dot(axes=(2, 1))([phi_x_matrix, pnet_output]) + last_layer_bias
        return tf.cast(u, variable_dtype)  #, tf.cast(phi_x, variable_dtype)

