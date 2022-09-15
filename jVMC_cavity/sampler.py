import jax
import jax.numpy as jnp
import numpy as np
import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs


class ExactSampler_LC:
    """Class for full enumeration of basis states in the lattice and cavity setting.

    This class generates a full basis of the many-body Hilbert space. Thereby, it \
    allows to exactly perform sums over the full Hilbert space instead of stochastic \
    sampling. Dimensions of the local Hilbert spaces for the lattice and cavity parts \
    can be different.

    Initialization arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of computational basis states. For lattice of size L \
        and the cavity, it should read (L + 1,).
        * ``inputDimLattice``: Local dimension of the lattice degree of freedom
        * ``inputDimCavity``: Local dimension of the cavity degree of freedom
        * ``logProbFactor``: Prefactor of the log-probability. Should be 0.5 for NQS wavefunction and 1.0 for POVM.
    """

    def __init__(self, net, sampleShape, inputDimLattice, inputDimCavity, logProbFactor=1.0):
        self.psi = net
        self.N = jnp.prod(jnp.asarray(sampleShape))
        self.sampleShape = sampleShape
        self.inputDimLattice = inputDimLattice
        self.inputDimCavity = inputDimCavity
        self.logProbFactor = logProbFactor

        # pmap'd member functions
        self._get_basis_pmapd = global_defs.pmap_for_my_devices(self._get_basis, in_axes=(0, 0, None, None),
                                                                static_broadcasted_argnums=(2, 3, 4))
        self._compute_probabilities_pmapd = global_defs.pmap_for_my_devices(self._compute_probabilities,
                                                                            in_axes=(0, None, 0))
        self._normalize_pmapd = global_defs.pmap_for_my_devices(self._normalize, in_axes=(0, None))

        self.get_basis()

        # Make sure that net params are initialized
        self.psi(self.basis)

        self.lastNorm = 0.

    def get_basis(self):
        myNumStates = mpi.distribute_sampling(self.inputDimCavity * (self.inputDimLattice ** (self.N - 1)))
        myFirstState = mpi.first_sample_id()

        deviceCount = global_defs.device_count()

        self.numStatesPerDevice = [(myNumStates + deviceCount - 1) // deviceCount] * deviceCount
        self.numStatesPerDevice[-1] += myNumStates - deviceCount * self.numStatesPerDevice[0]
        self.numStatesPerDevice = jnp.array(self.numStatesPerDevice)

        totalNumStates = deviceCount * self.numStatesPerDevice[0]

        intReps = jnp.arange(myFirstState, myFirstState + totalNumStates)
        intReps = intReps.reshape(
            (global_defs.device_count(), -1))  # all states are enumerated in an integer representation
        # and grouped on devices
        self.basis = jnp.zeros(intReps.shape + (self.N,), dtype=np.int32)
        self.basis = self._get_basis_pmapd(self.basis, intReps, self.inputDimLattice, self.inputDimCavity,
                                           self.sampleShape)

    def _get_basis(self, states, intReps, inputDimLattice, inputDimCavity, sampleShape):
        def make_state(state, intRep):
            intRepCavity = intRep % inputDimCavity
            intRepLattice = (intRep - intRepCavity) // inputDimCavity

            def scan_fun(c, x):  # translates integer representation of state into a (list) state made of configurations
                locState = c % inputDimLattice
                c = (c - locState) // inputDimLattice
                return c, locState

            _, stateLattice = jax.lax.scan(scan_fun, intRepLattice, state[1:])
            state = state.at[0].set(intRepCavity).at[1:].set(stateLattice)
            return state[::-1].reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0, 0))(states, intReps)

        return basis

    def _compute_probabilities(self, logPsi, lastNorm, numStates):
        p = jnp.exp(jnp.real(logPsi - lastNorm) / self.logProbFactor)

        def scan_fun(c, x):
            out = jax.lax.cond(c[1] < c[0], lambda x: x[0], lambda x: x[1], (x, 0.))
            newC = c[1] + 1
            return (c[0], newC), out

        _, p = jax.lax.scan(scan_fun, (numStates, 0), p)

        return p

    def _normalize(self, p, nrm):
        return p / nrm

    def sample(self, parameters=None, numSamples=None, multipleOf=None):
        """Return all computational basis states.

        Sampling is automatically distributed across MPI processes and available \
        devices.

        Arguments:
            * ``parameters``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.
            * ``numSamples``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.
            * ``multipleOf``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.

        Returns:
            ``configs, logPsi, p``: All computational basis configurations, \
            corresponding wave function coefficients, and probabilities \
            :math:`|\psi(s)|^2` (normalized).
        """

        logPsi = self.psi(self.basis)

        p = self._compute_probabilities_pmapd(logPsi, self.lastNorm, self.numStatesPerDevice)

        nrm = mpi.global_sum(p)
        p = self._normalize_pmapd(p, nrm)

        self.lastNorm += self.logProbFactor * jnp.log(nrm)

        return self.basis, logPsi, p

    def set_number_of_samples(self, N):
        pass

# ** end class ExactSampler
