import jax
from jax.config import config

config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets.initializers import init_fn_args
from jVMC.util.symmetries import LatticeSymmetry
from jVMC.nets.rnn1d_general import RNNCellStack, RNNCell, LSTMCell, GRUCell

from typing import Union
import warnings

from functools import partial

from typing import Sequence


class MultiLayerPerceptron(nn.Module):
    features: Sequence[int]
    actFun: callable
    init_args: dict

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.actFun(nn.Dense(features=feat, **self.init_args)(x))

        x = nn.Dense(features=self.features[-1],
                     use_bias=True,
                     **self.init_args,
                     name='last_layer')(x)
        return x


class RNN1DGeneral_LC(nn.Module):
    """
    Implementation of a multi-layer RNN for one-dimensional data with arbitrary cell.

    The ``cell`` parameter can be a string ("RNN", "LSTM", or "GRU") indicating a pre-implemented
    cell. Alternatively, a custom cell can be passed in the form of a tuple containing a flax
    module that implements the hidden state update rule and the initial value of the hidden state
    (i.e., the initial ``carry``).
    The signature of the ``__call__`` function of the cell flax module has to be
    ``(carry, state) -> (new_carry, output)``.

    This model can produce real positive or complex valued output. In either case the output is
    normalized such that

        :math:`\sum_s |RNN(s)|^{1/\kappa}=1`.

    Here, :math:`\kappa` corresponds to the initialization parameter ``logProbFactor``. Thereby, the RNN
    can represent both probability distributions and wave functions. Real or complex valued output is
    chosen through the parameter ``realValuedOutput``.

    The RNN allows for autoregressive sampling through the ``sample`` member function.

    Initialization arguments:
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDimLattice``: dimension of the local configuration space for lattice
        * ``inputDimCavity``: dimension of the local configuration space for cavity
        * ``denseCavityLayers``: Tuple of layer sizes of the multiperceptron whose output is the cavity state, not including the last layer that is constructed automatically. For a single layer with linear activation, provide (None,).
        * ``actFun``: non-linear activation function for the RNN cells
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``realValuedOutput``: Boolean indicating whether the network output is a real or complex number.
        * ``realValuedParams``: Boolean indicating whether the network parameters are real or complex parameters.
        * ``layersCavity``: tuple with entries corresponding to sizes of the multilayer perceptron layer in the cavity part of the neural network. Provide layersCavity = (10, 10, ) for two layers with size 10        followed by a dense layer with linear activation or layersCavity = (None, ) if only one dense layer with linear activation is to be applied.
        * ``cell``: String ("RNN", "LSTM", or "GRU") or custom definition indicating which type of cell to use for hidden state transformations.

    """

    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDimLattice: int = 2
    actFun: callable = nn.elu
    inputDimCavity: int = 2
    denseCavityLayers: tuple = (None, )

    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"

    def setup(self):
        if isinstance(self.cell, str) and self.cell != "RNN":
            ValueError("Complex parameters for LSTM/GRU not yet implemented.")

        if self.realValuedParams:
            self.dtype = global_defs.tReal
            self.initFunction = jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_avg",
                                                                     distribution="uniform")
        else:
            warnings.warn("Complex values for the RNN implemented but not tested.")
            self.dtype = global_defs.tCpx
            self.initFunction = partial(jVMC.nets.initializers.cplx_variance_scaling, scale=self.initScale)

        if isinstance(self.cell, str):
            self.zero_carry = jnp.zeros((self.depth, 1, self.hiddenSize), dtype=self.dtype)
            if self.cell == "RNN":
                self.cells = [RNNCell(actFun=self.actFun, initFun=self.initFunction, dtype=self.dtype) for _ in
                              range(self.depth)]
            elif self.cell == "LSTM":
                self.cells = [LSTMCell() for _ in range(self.depth)]
                self.zero_carry = jnp.zeros((self.depth, 2, self.hiddenSize), dtype=self.dtype)
            elif self.cell == "GRU":
                self.cells = [GRUCell() for _ in range(self.depth)]
            else:
                ValueError("Cell name not recognized.")
        else:
            self.cells = self.cell[0]
            self.zero_carry = self.cell[1]

        self.rnnCell = RNNCellStack(self.cells, actFun=self.actFun)
        init_args = init_fn_args(dtype=self.dtype, bias_init=jax.nn.initializers.zeros, kernel_init=self.initFunction)

        # CHANGED FROM jVMC - there was features=(self.inputDim - 1)*...
        self.outputDenseLattice = MultiLayerPerceptron(features=[self.inputDimLattice * (2 - self.realValuedOutput)],
                                                          actFun=self.actFun, init_args=init_args)

        if self.denseCavityLayers[0] is None:
            self.outputDenseCavity = MultiLayerPerceptron(features=[self.inputDimCavity * (2 - self.realValuedOutput)],
                                                          actFun=self.actFun, init_args=init_args)
        else:
            self.outputDenseCavity = MultiLayerPerceptron(features=[*self.denseCavityLayers,
                                                                    self.inputDimCavity * (2 - self.realValuedOutput)],
                                                          actFun=self.actFun, init_args=init_args)


    def log_coeffs_to_log_probs(self, logCoeffs, inputDim):
        phase = jnp.zeros(inputDim)

        if not self.realValuedOutput and self.realValuedParams:
            phase = 1.j * logCoeffs[inputDim:]
        amp = logCoeffs[:inputDim]

        return (self.logProbFactor * jax.nn.log_softmax(amp)) + phase

    def __call__(self, x):

        (stateLattice, _), probsLattice = self.rnn_cell((self.zero_carry, jnp.zeros(self.inputDimLattice)),
                                                                jax.nn.one_hot(x[:-1], self.inputDimLattice))
        # produce hidden state for input configuration of the last lattice site
        # this step is needed, otherwise probsCavity does not depend on the configuration of the last lattice site

        stateLattice, _ = self.rnnCell(stateLattice, jax.nn.one_hot(x[-2], self.inputDimLattice))

        # if we deal with eg. LSTM, stateLattice contains both hidden state stateLattice[:, 1, :]
        # and cell state stateLattice[:, 0, :]. Only the hidden state shall be used for the dense_cavity part.

        probsCavity = self.dense_cavity(self.get_hidden_state(stateLattice),
                                        jax.nn.one_hot(x[-1:], self.inputDimCavity))

        return jnp.sum(probsLattice, axis=0) + jnp.sum(probsCavity, axis=0)

    def get_hidden_state(self, stateLattice):
        if self.zero_carry.shape[1] == 2:
            return stateLattice[:, 1, :].reshape(self.hiddenSize * self.depth)
        else:
            return stateLattice[:, 0, :].reshape(self.hiddenSize * self.depth)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logProb = self.log_coeffs_to_log_probs(self.outputDenseLattice(out), self.inputDimLattice)
        logProb = jnp.sum(logProb * x, axis=-1)  # chooses one sigma_j component
                                                 # from inputDimLattice components of log P(sigma_j|sigma_1...sigma_j-1)
        return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)

    def dense_cavity(self, stateLatticeHidden, x):
        logProb = self.log_coeffs_to_log_probs(self.outputDenseCavity(stateLatticeHidden),
                                               self.inputDimCavity)
        logProb = jnp.sum(logProb * x, axis=-1)

        return jnp.nan_to_num(logProb, nan=-35)

    def sample(self, batchSize, key):
        def generate_sample(key):
            myKeys = jax.random.split(key, self.L + 1)
            myKeysLattice = myKeys[:-1]
            myKeysCavity = myKeys[-1]

            (stateLattice, _), (logProbsLattice, configLattice) = self.rnn_cell_sample(
                (self.zero_carry, jnp.zeros(self.inputDimLattice)),
                (myKeysLattice)
            )
            stateLattice, _ = self.rnnCell(stateLattice, jax.nn.one_hot(configLattice[-2], self.inputDimLattice))
            # this last step is needed, otherwise cavity does not depend on the configuration of the last site
            _, configCavity = self.cavity_sample(self.get_hidden_state(stateLattice), myKeysCavity)
            configs = jnp.concatenate((configLattice, configCavity.reshape((1,))), axis=0)
            return jnp.transpose(configs)

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)

    @partial(nn.transforms.scan,  # scan is over keys
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sample(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])

        logProbs = self.log_coeffs_to_log_probs(self.outputDenseLattice(out), self.inputDimLattice)
        sampleOut = jax.random.categorical(x, jnp.real(logProbs) / self.logProbFactor)
        sample = jax.nn.one_hot(sampleOut, self.inputDimLattice)
        logProb = jnp.sum(logProbs * sample)  # probability of having chosen the chosen configuration
        return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)

    def cavity_sample(self, stateLatticeHidden, key):
        y = self.outputDenseCavity(stateLatticeHidden)
        logProbs = self.log_coeffs_to_log_probs(y, self.inputDimCavity)

        sampleOut = jax.random.categorical(key, jnp.real(logProbs) / self.logProbFactor)
        sample = jax.nn.one_hot(sampleOut, self.inputDimCavity)
        logProb = jnp.sum(logProbs * sample)

        return jnp.nan_to_num(logProb, nan=-35), sampleOut


class RNN1DGeneral_LCSym(nn.Module):
    """
    Implementation of an RNN which consists of an RNNCellStack with an additional output layer.
    It uses the RNN class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Initialization arguments:
        * ``orbit``: collection of maps that define symmetries (instance of ``util.symmetries.LatticeSymmetry``)
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDimLattice``: dimension of the local configuration space for lattice
        * ``inputDimCavity``: dimension of the local configuration space for cavity
        * ``actFun``: non-linear activation function for the RNN cells
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``realValuedOutput``: Boolean indicating whether the network output is a real or complex number.
        * ``realValuedParams``: Boolean indicating whether the network parameters are real or complex parameters.
        * ``cell``: String ("RNN", "LSTM", or "GRU") or custom definition indicating which type of cell to use for hidden state transformations.

    """
    orbit: LatticeSymmetry
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDimLattice: int = 2
    actFun: callable = nn.elu
    inputDimCavity: int = 2
    denseCavityLayers: tuple = (None,)


    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"

    def setup(self):

        self.rnn = RNN1DGeneral_LC(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                                inputDimLattice=self.inputDimLattice,
                                actFun=self.actFun, inputDimCavity=self.inputDimCavity,
                                denseCavityLayers=self.denseCavityLayers,
                                initScale=self.initScale,
                                logProbFactor=self.logProbFactor,
                                realValuedOutput=self.realValuedOutput,
                                realValuedParams=self.realValuedParams,
                                cell=self.cell)  # bug in jVMC?

    def __call__(self, x):

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jnp.exp((1. / self.logProbFactor) * jax.vmap(evaluate)(x)), axis=0)

        logProbs = self.logProbFactor * jnp.log(res)

        return logProbs

    def sample(self, batchSize, key):

        key1, key2 = jax.random.split(key)

        configs = self.rnn.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, self.orbit.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs

# ** end class RNN1DGeneral_LCSym


class RNN1DGeneral_LC2(nn.Module):
    """
    Implementation of a multi-layer RNN for one-dimensional data with arbitrary cell.

    The ``cell`` parameter can be a string ("RNN", "LSTM", or "GRU") indicating a pre-implemented
    cell. Alternatively, a custom cell can be passed in the form of a tuple containing a flax
    module that implements the hidden state update rule and the initial value of the hidden state
    (i.e., the initial ``carry``).
    The signature of the ``__call__`` function of the cell flax module has to be
    ``(carry, state) -> (new_carry, output)``.

    This model can produce real positive or complex valued output. In either case the output is
    normalized such that

        :math:`\sum_s |RNN(s)|^{1/\kappa}=1`.

    Here, :math:`\kappa` corresponds to the initialization parameter ``logProbFactor``. Thereby, the RNN
    can represent both probability distributions and wave functions. Real or complex valued output is
    chosen through the parameter ``realValuedOutput``.

    The RNN allows for autoregressive sampling through the ``sample`` member function.

    Initialization arguments:
        * ``L``: length of the spin chain
        * ``hiddenSizeLattice``: size of the hidden state vector, lattice part
        * ``hiddenSizeCavity``: size of the hidden state vector, cavity part
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDimLattice``: dimension of the local configuration space for lattice
        * ``inputDimCavity``: dimension of the local configuration space for cavity
        * ``actFun``: non-linear activation function for the RNN cells
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``realValuedOutput``: Boolean indicating whether the network output is a real or complex number.
        * ``realValuedParams``: Boolean indicating whether the network parameters are real or complex parameters.
        * ``cell``: String ("RNN", "LSTM", or "GRU") or custom definition indicating which type of cell to use for hidden state transformations.

    """

    L: int = 10
    hiddenSizeLattice: int = 10
    hiddenSizeCavity: int = 10
    depth: int = 1
    inputDimLattice: int = 2
    actFun: callable = nn.elu
    inputDimCavity: int = 2

    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"

    def setup(self):
        hiddenSize = self.hiddenSizeLattice + self.hiddenSizeCavity

        if isinstance(self.cell, str) and self.cell != "RNN":
            ValueError("Complex parameters for LSTM/GRU not yet implemented.")

        if self.realValuedParams:
            self.dtype = global_defs.tReal
            self.initFunction = jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_avg",
                                                                     distribution="uniform")
        else:
            warnings.warn("Complex values for the RNN implemented but not tested.")
            self.dtype = global_defs.tCpx
            self.initFunction = partial(jVMC.nets.initializers.cplx_variance_scaling, scale=self.initScale)

        if isinstance(self.cell, str):
            self.zero_carry = jnp.zeros((self.depth, 1, hiddenSize), dtype=self.dtype)
            if self.cell == "RNN":
                self.cells = [RNNCell(actFun=self.actFun, initFun=self.initFunction, dtype=self.dtype) for _ in
                              range(self.depth)]
            elif self.cell == "LSTM":
                self.cells = [LSTMCell() for _ in range(self.depth)]
                self.zero_carry = jnp.zeros((self.depth, 2, hiddenSize), dtype=self.dtype)
            elif self.cell == "GRU":
                self.cells = [GRUCell() for _ in range(self.depth)]
            else:
                ValueError("Cell name not recognized.")
        else:
            self.cells = self.cell[0]
            self.zero_carry = self.cell[1]

        self.rnnCell = RNNCellStack(self.cells, actFun=self.actFun)
        init_args = init_fn_args(dtype=self.dtype, bias_init=jax.nn.initializers.zeros, kernel_init=self.initFunction)

        # CHANGED FROM jVMC - there was features=(self.inputDim - 1)*...
        self.outputDenseLattice = MultiLayerPerceptron(features=[self.inputDimLattice * (2 - self.realValuedOutput)],
                                                          actFun=self.actFun, init_args=init_args)


        self.outputDenseCavity = MultiLayerPerceptron(features=[self.inputDimCavity * (2 - self.realValuedOutput)],
                                                          actFun=self.actFun, init_args=init_args)


    def log_coeffs_to_log_probs(self, logCoeffs, inputDim):
        phase = jnp.zeros(inputDim)

        if not self.realValuedOutput and self.realValuedParams:
            phase = 1.j * logCoeffs[inputDim:]
        amp = logCoeffs[:inputDim]

        return (self.logProbFactor * jax.nn.log_softmax(amp)) + phase

    def __call__(self, x):

        (stateLattice, _), probsLattice = self.rnn_cell((self.zero_carry, jnp.zeros(self.inputDimLattice)),
                                                                jax.nn.one_hot(x[:-1], self.inputDimLattice))
        # produce hidden state for input configuration of the last lattice site
        # this step is needed, otherwise probsCavity does not depend on the configuration of the last lattice site

        stateLattice, _ = self.rnnCell(stateLattice, jax.nn.one_hot(x[-2], self.inputDimLattice))

        # if we deal with eg. LSTM, stateLattice contains both hidden state stateLattice[:, 1, :]
        # and cell state stateLattice[:, 0, :]. Only the hidden state shall be used for the dense_cavity part.

        probsCavity = self.dense_cavity(self.get_hidden_state_cavity(stateLattice),
                                        jax.nn.one_hot(x[-1:], self.inputDimCavity))

        return jnp.sum(probsLattice, axis=0) + jnp.sum(probsCavity, axis=0)

    def get_hidden_state_cavity(self, stateLattice):
        # takes the cavity part of the hidden state i.e. its last self.hiddenSizeCavity components.
        if self.zero_carry.shape[1] == 2:
            return stateLattice[:, 1, self.hiddenSizeLattice:].reshape(self.hiddenSizeCavity * self.depth)
        else:
            return stateLattice[:, 0, self.hiddenSizeLattice:].reshape(self.hiddenSizeCavity * self.depth)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        # Uses only the first self.hiddenSizeLattice components of the hidden state for lattice dof probabilities.
        # The rest is used for the cavity mode.

        logProb = self.log_coeffs_to_log_probs(self.outputDenseLattice(out[:self.hiddenSizeLattice]), self.inputDimLattice)
        logProb = jnp.sum(logProb * x, axis=-1)  # chooses one sigma_j component
                                                 # from inputDimLattice components of log P(sigma_j|sigma_1...sigma_j-1)
        return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)

    def dense_cavity(self, state_hidden, x):
        logProb = self.log_coeffs_to_log_probs(self.outputDenseCavity(state_hidden),
                                               self.inputDimCavity)
        logProb = jnp.sum(logProb * x, axis=-1)

        return jnp.nan_to_num(logProb, nan=-35)

    def sample(self, batchSize, key):
        def generate_sample(key):
            myKeys = jax.random.split(key, self.L + 1)
            myKeysLattice = myKeys[:-1]
            myKeysCavity = myKeys[-1]

            (stateLattice, _), (logProbsLattice, configLattice) = self.rnn_cell_sample(
                (self.zero_carry, jnp.zeros(self.inputDimLattice)),
                (myKeysLattice)
            )
            stateLattice, _ = self.rnnCell(stateLattice, jax.nn.one_hot(configLattice[-2], self.inputDimLattice))
            # this last step is needed, otherwise cavity does not depend on the configuration of the last site
            _, configCavity = self.cavity_sample(self.get_hidden_state_cavity(stateLattice), myKeysCavity)
            configs = jnp.concatenate((configLattice, configCavity.reshape((1,))), axis=0)
            return jnp.transpose(configs)

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)

    @partial(nn.transforms.scan,  # scan is over keys
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sample(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])

        logProbs = self.log_coeffs_to_log_probs(self.outputDenseLattice(out[:self.hiddenSizeLattice]), self.inputDimLattice)
        sampleOut = jax.random.categorical(x, jnp.real(logProbs) / self.logProbFactor)
        sample = jax.nn.one_hot(sampleOut, self.inputDimLattice)
        logProb = jnp.sum(logProbs * sample)  # probability of having chosen the chosen configuration
        return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)

    def cavity_sample(self, stateLatticeHidden, key):
        y = self.outputDenseCavity(stateLatticeHidden)
        logProbs = self.log_coeffs_to_log_probs(y, self.inputDimCavity)

        sampleOut = jax.random.categorical(key, jnp.real(logProbs) / self.logProbFactor)
        sample = jax.nn.one_hot(sampleOut, self.inputDimCavity)
        logProb = jnp.sum(logProbs * sample)

        return jnp.nan_to_num(logProb, nan=-35), sampleOut


class RNN1DGeneral_LC2Sym(nn.Module):
    """
    Implementation of an RNN which consists of an RNNCellStack with an additional output layer.
    It uses the RNN class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Initialization arguments:
        * ``orbit``: collection of maps that define symmetries (instance of ``util.symmetries.LatticeSymmetry``)
        * ``L``: length of the spin chain
        * ``hiddenSizeLattice``: size of the hidden state vector, lattice part
        * ``hiddenSizeCavity``: size of the hidden state vector, cavity part
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDimLattice``: dimension of the local configuration space for lattice
        * ``inputDimCavity``: dimension of the local configuration space for cavity
        * ``actFun``: non-linear activation function for the RNN cells
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``realValuedOutput``: Boolean indicating whether the network output is a real or complex number.
        * ``realValuedParams``: Boolean indicating whether the network parameters are real or complex parameters.
        * ``cell``: String ("RNN", "LSTM", or "GRU") or custom definition indicating which type of cell to use for hidden state transformations.

    """
    orbit: LatticeSymmetry
    L: int = 10
    hiddenSizeLattice: int = 10
    hiddenSizeCavity: int = 10
    depth: int = 1
    inputDimLattice: int = 2
    actFun: callable = nn.elu
    inputDimCavity: int = 2


    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"

    def setup(self):

        self.rnn = RNN1DGeneral_LC2(L=self.L, hiddenSizeLattice=self.hiddenSizeLattice,
                                    hiddenSizeCavity=self.hiddenSizeCavity,
                                    depth=self.depth,
                                    inputDimLattice=self.inputDimLattice,
                                    actFun=self.actFun, inputDimCavity=self.inputDimCavity,
                                    initScale=self.initScale,
                                    logProbFactor=self.logProbFactor,
                                    realValuedOutput=self.realValuedOutput,
                                    realValuedParams=self.realValuedParams,
                                    cell=self.cell)  # bug in jVMC?

    def __call__(self, x):

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jnp.exp((1. / self.logProbFactor) * jax.vmap(evaluate)(x)), axis=0)

        logProbs = self.logProbFactor * jnp.log(res)

        return logProbs

    def sample(self, batchSize, key):

        key1, key2 = jax.random.split(key)

        configs = self.rnn.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, self.orbit.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        return configs

# ** end class RNN1DGeneral_LC2Sym