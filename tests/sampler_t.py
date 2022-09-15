import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
import jVMC
from functools import partial
import numpy as np


from jVMC_cavity.povm import POVM_LC, POVMOperator_LC, measure_povm_LC
from jVMC_cavity.sampler import ExactSampler_LC
from jVMC_cavity.nets.rnn1d_general import RNN1DGeneral_LC
from jVMC_cavity.utils import copy_dict, normalize, norm_fun, set_initial_state


def state_to_int(s, inputDimLattice):
    def for_fun(i, xs):
        return xs[0] + xs[1][i] * (inputDimLattice ** i), xs[1]

    x, _ = jax.lax.fori_loop(0, s.shape[-1], for_fun, (0, s))

    return x


class TestSampler(unittest.TestCase):

    def test_autoregressive_sampling_with_rnn(self):
        L = 2
        sample_shape = (L + 1,)
        logProbFactor = 1
        inputDimLattice = 2
        inputDimCavity = 3  # number of Fock states
        numSamples = 1000000

        initial_lattice_state_vector = normalize(jnp.array([1, 1]))
        initial_cavity_state_vector = normalize(jnp.array([1, 0, 0]))

        # Initialize net
        sample_shape = (L + 1,)
        net_kwargs = dict(L=L,
                          hiddenSize=2,
                          depth=2,
                          inputDimLattice=inputDimLattice ** 2,  # for SIC-POVM
                          actFun=nn.elu,
                          inputDimCavity=inputDimCavity ** 2,  # for SIC-POVM
                          initScale=1,
                          logProbFactor=logProbFactor,
                          realValuedOutput=True,
                          realValuedParams=True,
                          cell="RNN")

        net = RNN1DGeneral_LC(**net_kwargs)

        psi = jVMC.vqs.NQS(net, batchSize=1000, seed=1234)
        # to compile one has to evaluate once with a certain dimensional input
        psi(jnp.zeros((jVMC.global_defs.device_count(), 1) + sample_shape, dtype=jnp.int32))

        povm = POVM_LC(L,
                       inputDimCavity=inputDimCavity,
                       inputDimLattice=inputDimLattice,
                       maxCorrLength=0)

        set_initial_state(psi, povm, initial_lattice_state_vector, subspace="lattice")
        set_initial_state(psi, povm, initial_cavity_state_vector, subspace="cavity")

        # Set up exact sampler
        exactSampler = ExactSampler_LC(psi, sample_shape,
                                                       inputDimLattice=inputDimLattice**2,  # because SIC-POVM
                                                       inputDimCavity=inputDimCavity**2,  # because SIC-POVM
                                                       logProbFactor=logProbFactor)
        # Set up MCMC sampler
        mcSampler = jVMC.sampler.MCSampler(psi, (L + 1,), random.PRNGKey(123))

        # Compute exact probabilities
        se, logpe, pe = exactSampler.sample()

        self.assertTrue(se.shape[1] == inputDimLattice**(2*L) * inputDimCavity**2)

        # Perform autoregressive sampling
        smc, logp, _ = mcSampler.sample(numSamples=numSamples)

        self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logp))) < 1e-12)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))
        se = se.reshape((se.shape[0] * se.shape[1], -1))


        seIntRep = jax.vmap(partial(state_to_int, inputDimLattice=inputDimLattice ** 2))(se)
        self.assertTrue(jnp.unique(seIntRep).shape[0] == seIntRep.shape[0])

        seIntRepArgsort = jnp.argsort(seIntRep)

        # Compute histogram of sampled configurations
        smcIntRep = jax.vmap(partial(state_to_int, inputDimLattice=inputDimLattice ** 2))(smc)

        pmc, _ = jnp.histogram(smcIntRep, bins=jnp.arange(0, se.shape[0] + 1), density=True)
        #print(jnp.max(jnp.abs(pmc - pe.ravel()[seIntRepArgsort])))
        self.assertTrue(jnp.max(jnp.abs(pmc - pe.ravel()[seIntRepArgsort])) < 1e-3)


if __name__ == "__main__":
    unittest.main()