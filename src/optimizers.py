import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
from tensorflow.python.keras import backend_config

import tensorflow_addons as tfa
from tensorflow_addons.optimizers import DecoupledWeightDecayExtension


class Adam_meta(keras.optimizers.Optimizer):
    '''
    Adam optimizer with `meta` parameter

    PARAMETERS:
    - meta:   meta-plasticity parameter, for now only allows scalar values (in paper allows layer-wise)

    NOTE:
        the rest parameters are similar to original Adam. But got rid of `decay`

    TODO:
    - consider adding `f_meta` as an option
    - consider applying heterogeneity in `meta` like paper

    SOURCE:
    - [Adam-keras](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/adam.py)
    - [AdamW-keras](https://github.com/OverLordGoldDragon/keras-adamw/blob/master/keras_adamw/optimizers_v2.py)
    - [Adam_meta-Torch](https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN/blob/master/Continual_Learning_Fig-2abcdefgh-3abcd-5cde/models_utils.py)
    - [DecoupledWeightDecayExtension]: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/DecoupledWeightDecayExtension

    '''

    def __init__(self,
                 meta=0,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 amsgrad=False,
                 name="Adam-meta",
                 **kwargs):
        # Check for conditions
        if min(meta, learning_rate, epsilon) < 0.0:
            raise ValueError('Invalid "meta" or "learning_rate" or "epsilon". Needs all to be non-negative')
        if min(beta_1, beta_2) < 0.0 or max(beta_1, beta_2) > 1.0:
            raise ValueError('Invalid "beta_1" or "beta_2". Needs both to be within [0,1]')

        # Initialization and add hyperparameters
        super(Adam_meta, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('meta', meta)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        '''Create slots for the first and second moments.
        Exactly similar to [AdamW-keras]
        '''
        for var in var_list:
            self.add_slot(var, 'm') # 1st moment
        for var in var_list:
            self.add_slot(var, 'v') # 2nd moment
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat') # 2nd moment in case AMSGrad
        self._updates_per_iter = len(var_list)


    @tf.function
    def _resource_apply_dense(self, grad, var):
        '''Update the slots and perform one optimization step for one model variable for metaplasticity
        This is mirroring [AdamW-keras] and [Adam_meta-Torch].
        '''
        var_device, var_dtype = var.device, var.dtype.base_dtype

        # Get slots for 1st and 2nd moments
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Get hyperparameters
        meta_t = array_ops.identity(self._get_hyper('meta', var_dtype))
        lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        # Compute parameters based on current local step
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step) # B1^t
        beta_2_power = math_ops.pow(beta_2_t, local_step) # B2^t

        # Learning rate bias correction
        # eta_t <- eta * sqrt(1-B2^t) / (1-B1^t)
        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Essential ADAM equations
        # m: 1st moment
        # v: 2nd moment
        # g: grad
        # m <- B1 * m + (1 - B1) * g
        # v <- B2 * v + (1 - B2) * g^2
        m_t = state_ops.assign(m, beta_1_t * m + (1.0 - beta_1_t) * grad, use_locking=self._use_locking)
        v_t = state_ops.assign(v, beta_2_t * v + (1.0 - beta_2_t) * math_ops.square(grad), use_locking=self._use_locking)

        # Apply AMSGrad if turned on
        # usually var_delta = dX <- m / (sqrt(v or v_hat) + eps)
        # var_delta_denom <- sqrt(v or v_hat) + eps
        # but metaplast will change a bit so only calc denom now
        if self.amsgrad: # v_hat <- max(v_hat, v_t)
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat, math_ops.maximum(vhat, v_t), use_locking=self._use_locking)
            var_delta_denom = math_ops.sqrt(vhat_t) + epsilon_t
        else:
            var_delta_denom = math_ops.sqrt(v_t) + epsilon_t

        # Metaplasticity
        if len(var.shape) == 1:  # True if bias or BN params, false if weight. TODO: Need to double check
            # X <- X - eta * sqrt(1-B2^t) / (1-B1^t) * m / (sqrt(v or v_hat) + eps)
            # X <- X - eta_t * dX
            # dX <- m / (var_delta_denom = sqrt(v or v_hat) + eps)
            var_t = math_ops.sub(var, lr_t * m_t / var_delta_denom)
        else:
            # the variables will be similar to [Adam_meta-Torch] code and try to mirror paper
            # binary_weight_before_update: Wb <- sign(Wh)
            # condition_consolidation: use_meta <- Uw * Wb > 0.0
            # Uw <- dX
            # Wh <- var
            Wb = math_ops.sign(var)
            use_meta = math_ops.multiply(Wb, m_t) > 0.0 # sign(m_t) = sign(dX)

            # f_meta = 1 - tanh(m * Wh)^2
            f_meta = array_ops.ones_like(var) - math_ops.square(math_ops.tanh(meta_t * var))

            # only use meta-applied m_t when use_meta = True
            # i.e. only use f_meta when Wb * m_t/denom > 0
            decayed_m_t = math_ops.multiply(f_meta, m_t)
            alt_m_t = array_ops.where(use_meta, decayed_m_t, m_t)

            # X <- X - eta_t * dX
            # dX <- (f_meta(X=Wh) if Wb*Uw >0 else 1.0) * m / (var_delta_denom = sqrt(v or v_hat) + eps)
            var_t = math_ops.sub(var, lr_t * alt_m_t / var_delta_denom)

        # Return updates
        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'meta': self._serialize_hyperparameter('meta'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        })
        return config


class Adam_meta_W(DecoupledWeightDecayExtension, Adam_meta):
    def __init__(self,
                 weight_decay,
                 meta           = 0,
                 learning_rate  = 0.001,
                 beta_1         = 0.9,
                 beta_2         = 0.999,
                 epsilon        = 1e-8,
                 amsgrad        = False,
                 name           = "Adam_meta_W",
                 **kwargs):
        super().__init__(
            weight_decay,
            meta            = meta,
            learning_rate   = learning_rate,
            beta_1          = beta_1,
            beta_2          = beta_2,
            epsilon         = epsilon,
            amsgrad         = amsgrad,
            name            = name,
            **kwargs)