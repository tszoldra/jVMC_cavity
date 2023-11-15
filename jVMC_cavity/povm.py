from abc import ABC

import qbism
import qutip
import jax
import jax.numpy as jnp
import scipy  # jax.scipy.sqrtm not implemented on GPU

import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
import jVMC.operator

import functools

opDtype = global_defs.tReal


def measure_povm_LC(povm, sampler, sampleConfigs=None, probs=None, observables=None, max_number_sites=None):
    """For a set of sampled configurations, compute the associated expectation values
    for a given set of observables. If none is provided, all observables defined in ``povm`` are computed.

    Note that this function does not average over the lattice sites, unlike in original implementation of `jVMC`.

    Caution: may give observables inaccurate even up to 1e-2 part depending on the choice of SIC-POVM.
    For example, S_Z=1.006 was observed in a normalized state for one POVM, and S_Z=0.999 for another POVM in the same state.

    Args:
        * ``povm``: the povm that holds the jitted evaluation function
        * ``sampler``: the sampler used for the computation of expectation values
        * ``sampleConfigs``: optional, if configurations have already been generated
        * ``probs``: optional, the associated probabilities
        * ``observables``: optional, observables for which expectation values are computed
        * ``max_number_sites``: how many consecutive sites to consider for computation of local expectation value.\
                                For example, for translationally invariant system, ``max_number_sites=1`` is enough
                                and saves time by a factor of the system size.
                                The result is anyway averaged if the network has symmetry implemented.
                                For translation_by_2 symmetry, set ``max_number_sites=2``.
    """
    if sampleConfigs == None:
        sampleConfigs, sampleLogPsi, probs = sampler.sample()

    if observables == None:
        observables = povm.observables

    if max_number_sites is None:
        max_number_sites = povm.L

    result = {}

    for name, ops in observables.items():
        if 'lattice' in ops and not 'cavity' in ops:  # measure lattice - first records 0...(L-1)
            results = povm.evaluate_observable(ops['lattice'], sampleConfigs[:, :, :max_number_sites])
        elif 'cavity' in ops and not 'lattice' in ops:  # measure cavity - last record with index L
            results = povm.evaluate_observable(ops['cavity'], sampleConfigs[:, :, povm.L:])
        else:
            raise NotImplementedError('Two-body lattice-cavity observables not implemented.')

        result[name] = {"mean": [None] * results[0].shape[2],
                        "variance": [None] * results[0].shape[2],
                        "MC_error": [None] * results[0].shape[2]}

        for site in range(results[0].shape[2]):
            if probs is not None:
                result[name]["mean"][site] = jnp.array(mpi.global_mean(results[0][:, :, site][..., None], probs)[0])
                result[name]["variance"][site] = jnp.array(mpi.global_variance(results[0][:, :, site][..., None], probs)[0])
                result[name]["MC_error"][site] = jnp.array(result[name]["variance"][site] / jnp.sqrt(sampler.get_last_number_of_samples()))
            else:
                result[name]["mean"][site] = jnp.array(mpi.global_mean(results[0][:, :, site]))
                result[name]["variance"][site] = jnp.array(mpi.global_variance(results[0][:, :, site]))
                result[name]["MC_error"][site] = jnp.array(result[name]["variance"][site] / jnp.sqrt(sampler.get_last_number_of_samples()))

        if 'lattice' in ops and not 'cavity' in ops:
            # TODO implement the correlation calculation
            # two-point correlators are only measured for the lattice, not cavity
            for corrLen, corrVals in results[1].items():
                result_name = name + "_corr_L" + str(corrLen)

                result[result_name] = {"mean": [None] * max_number_sites,
                                       "variance": [None] * max_number_sites,
                                       "MC_error": [None] * max_number_sites}

                for site in range(max_number_sites):
                    if probs is not None:
                        result[result_name]["mean"][site] = jnp.array(mpi.global_mean(corrVals[:, :, site], probs) - result[name]["mean"][site]**2)
                        result[result_name]["variance"][site] = jnp.array(mpi.global_variance(corrVals[:, :, site], probs))
                        result[result_name]["MC_error"][site] = jnp.array(0.)
                    else:
                        result[result_name]["mean"][site] = jnp.array(mpi.global_mean(corrVals[:, :, site]) - result[name]["mean"][site]**2)
                        result[result_name]["variance"][site] = jnp.array(mpi.global_variance(corrVals[:, :, site]))
                        result[result_name]["MC_error"][site] = jnp.array(result[result_name]["variance"][site] / jnp.sqrt(sampler.get_last_number_of_samples()))

    for op_name, quantities in result.items():
        for quantity_name, quantity in quantities.items():
            result[op_name][quantity_name] = jnp.array(quantity)

    return result


def get_1_particle_distributions_LC(state_vector, povm, subspace: str):
    """Compute 1 particle POVM-representations, mainly used for defining initial states.

    Args:
        * ``state_vector``: the desired pure state vector in the photon subspace, as a numpy array
        * ``povm``: the povm for which the distribution is desired
        * ``subspace``: string equal to either 'lattice' or 'cavity' denoting the subspace with its own local dimension.
    """
    if subspace == 'cavity':
        M = povm.M_c
        probs = jnp.zeros(povm.inputDimCavity**2)
    elif subspace == 'lattice':
        M = povm.M_l
        probs = jnp.zeros(povm.inputDimLattice**2)
    else:
        raise ValueError("The subspace argument is neither 'cavity' nor 'lattice'.")

    for (idx, ele) in enumerate(M):
        probs = probs.at[idx].add(jnp.real(jnp.dot(jnp.conj(jnp.transpose(state_vector)), jnp.dot(ele, state_vector))))

    return probs


def get_M(d, ic_povm='symmetric'):
    """Returns ``d^2`` IC-POVM measurement operators for local Hilbert space dimension ``d``.

    Args:
        * ``d`` - local dimension of the local Hilbert space.
        For SIC-POVM must be less or equal ``151``.
        * ``ic_povm`` - type of the minimal, informationally complete POVM.
        Can be 'symmetric' for SIC-POVM or 'orthocross'. The latter is constructed fast for arbitrary dimension\
        and is supposedly better suited for dimensions ``d>2`` in Fock space with high occupancy of small\
        number of particles mode.

    Returns:
        jnp.array with the leading axis giving the different POVM-Measurement operators.

    """
    if d > 151 and ic_povm == 'symmetric':
        raise NotImplementedError('SIC-POVMs for dimension more than 151 are not implemented.')

    if ic_povm == 'symmetric':
        M = jnp.array([_op_from_qutip(q) for q in qbism.sic_povm(d)], dtype=global_defs.tCpx)
        return M
    elif ic_povm == 'orthocross':
        # implementation follows
        # J. DeBrota, PhD Thesis, Informationally Complete Measurements and Optimal
        # Representations of Quantum Theory, University of Massachusetts Boston, 2020
        # pages 30-31.
        # https://scholarworks.umb.edu/cgi/viewcontent.cgi?article=1616&context=doctoral_dissertations

        # first d elements are projectors |alpha><alpha|
        projectors_Pi = [qutip.basis(d, j).proj() for j in range(d)]

        # rest - off-diagonal coherence terms
        for k in range(d):
            for j in range(k):
                e_j = qutip.basis(d, j)
                e_k = qutip.basis(d, k)
                projectors_Pi.append(0.5 * (e_j + e_k) * (e_j.dag() + e_k.dag()))
                projectors_Pi.append(0.5 * (e_j + 1.0j * e_k) * (e_j.dag() - 1.0j * e_k.dag()))

        assert abs(len(projectors_Pi) - d**2) < 1e-12

        projectors_Pi = jnp.array([_op_from_qutip(op) for op in projectors_Pi])
        omega = jnp.asarray(jnp.sum(projectors_Pi, axis=0))
        omega_inv_sqrtm = jnp.array(scipy.linalg.inv(scipy.linalg.sqrtm(omega)), dtype=global_defs.tCpx)
        # workaround with scipy - jax.scipy.sqrtm not implemented.

        M = jnp.einsum('ij, ajk, kl -> ail', omega_inv_sqrtm, projectors_Pi, omega_inv_sqrtm)

        return M


def M_2Body(M1, M2):
    M = jnp.array([[jnp.kron(M1[i], M2[j]) for j in range(M2.shape[0])] for i in range(M1.shape[0])])
    return M.reshape((-1, M.shape[2], M.shape[3]))


def T_inv_2Body(T_inv1, T_inv2):
    return jnp.kron(T_inv1, T_inv2)


def get_D(L, Ldag, M, T_inv):
    return jnp.array(jnp.real(jnp.einsum('ij, bc, cjk, kl, ali -> ab', L, T_inv, M, Ldag, M)
                              - 0.5 * jnp.einsum('ij, jk, bc, ckl, ali -> ab', Ldag, L, T_inv, M, M)
                              - 0.5 * jnp.einsum('ij, jk, akl, bc, cli -> ab', Ldag, L, M, T_inv, M)),
                     dtype=opDtype)


def get_U(A, M, T_inv):
    return jnp.array(jnp.real(- 1.j * jnp.einsum('ij, bc, cjk, aki -> ab', A, T_inv, M, M)
                              + 1.j * jnp.einsum('ij, ajk, bc, cki -> ab', A, M, T_inv, M)), dtype=opDtype)



def get_Omega(A, M, T_inv):
    return jnp.array(jnp.real(jnp.einsum('ab, bij, ji -> a', T_inv, M, A)), dtype=opDtype)


def _op_from_qutip(qobj):
    return jnp.array(qobj.data.toarray(), dtype=global_defs.tCpx)


def get_paulis(inputDim):
    return tuple([_op_from_qutip(op) for op in qutip.operators.jmat((inputDim-1.)/2.)])


def aOp(lDim):
    """Returns a :math:`\hat a` bosonic annihilation operator matrix.

    Args:
    * ``lDim``: Local dimension of the Hilbert space (cutoff of the infinitely-dimensional space).

    Returns:
        Matrix form of :math:`\hat a` bosonic annihilation operator.
    """
    return _op_from_qutip(qutip.destroy(lDim))


def adagOp(lDim):
    """Returns a :math:`\hat a^\dagger` bosonic creation operator matrix.

    Args:
    * ``lDim``: Local dimension of the Hilbert space (cutoff of the infinitely-dimensional space).

    Returns:
        Matrix form of :math:`\hat a` bosonic creation operator.
    """
    return _op_from_qutip(qutip.create(lDim))


def nOp(lDim):
    """Returns a :math:`\hat n = \hat a^\dagger \hat a` bosonic number operator matrix.

    Args:
    * ``lDim``: Local dimension of the Hilbert space (cutoff of the infinitely-dimensional space).

    Returns:
        Matrix form of :math:`\hat n = \hat a^\dagger \hat a` photon number operator.
    """
    return _op_from_qutip(qutip.operators.num(lDim))



class POVM_LC():
    """This class provides SIC-POVM - operators and related matrices for lattice and cavity.

    Initializer arguments:
        * ``L``: length of the lattice
        * ``inputDimCavity``: size of the cavity local Hilbert space
        * ``inputDimLattice``: size of the lattice local Hilbert space
        * ``maxCorrLength``: maximal distance between lattice sites for correlation calculation
        * ``icPOVMCavity``: type of the minimal, informationally complete POVM for the cavity part.
        Can be 'symmetric' for SIC-POVM or 'orthocross'. The latter is constructed fast for arbitrary dimension\
        and is supposedly better suited for dimensions ``d>2`` in Fock space with high occupancy of small\
        number of particles mode.
        * ``icPOVMLattice``: type of the minimal, informationally complete POVM for the lattice part.
        Can be 'symmetric' for SIC-POVM or 'orthocross'. The latter is constructed fast for arbitrary dimension\
        and is supposedly better suited for dimensions ``d>2`` in Fock space with high occupancy of small\
        number of particles mode. For spins-1/2, choose 'symmetric'.
    """

    def __init__(self, L, inputDimCavity, inputDimLattice, maxCorrLength=0,
                 icPOVMCavity='symmetric',
                 icPOVMLattice='symmetric'):
        self.L = L
        self.inputDimCavity = inputDimCavity
        self.inputDimLattice = inputDimLattice
        self.maxCorrLength = maxCorrLength
        self.icPOVMCavity = icPOVMCavity
        self.icPOVMLattice = icPOVMLattice
        self.set_standard_povm_operators()
        if maxCorrLength > 0:
            raise NotImplementedError('Correlation calculation not implemented.')

        self._evaluate_magnetization_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda ops, idx:
                                                                                      self._evaluate_observable(ops, idx), in_axes=(None, 0)), in_axes=(None, 0))
        # TODO Implement and test correlation calculation.
        self._evaluate_correlators_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda resPerSamplePerSite, corrLen: resPerSamplePerSite *
                                                                                    jnp.roll(resPerSamplePerSite, corrLen, axis=0), in_axes=(0, None)), in_axes=(0, None))
        self._lattice_average_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda obsPerSamplePerSite: jnp.mean(obsPerSamplePerSite, axis=-1), in_axes=(0,)), in_axes=(0,))

    def set_standard_povm_operators(self):
        """
        Obtain matrices required for dynamics and observables.
        """
        self.M_l = get_M(self.inputDimLattice, ic_povm=self.icPOVMLattice)
        self.M_c = get_M(self.inputDimCavity, ic_povm=self.icPOVMCavity)

        def TT_inv(M1, M2):
            T = jnp.einsum('aij, bji -> ab', M1, M2)
            T_inv = jnp.linalg.inv(T)
            return T, T_inv

        self.T_ll, self.T_inv_ll = TT_inv(self.M_l, self.M_l)
        self.T_cc, self.T_inv_cc = TT_inv(self.M_c, self.M_c)

        self.dissipators = self._get_dissipators()

        self.unitaries = self._get_unitaries()

        self.operators = {**self.unitaries, **self.dissipators}
        self.observables = self._get_observables()

    def _get_dissipators(self):
        """Get the dissipation operators in the POVM-formalism.

        Returns:
            Dictionary with 1-body dissipation channels.
        """
        dissipators_POVM = {}

        #### Cavity
        a_c = aOp(self.inputDimCavity)
        adag_c = adagOp(self.inputDimCavity)
        sigmas_c = get_paulis(self.inputDimCavity)

        # Out of all matrix elements D(a,b) where a = a_lattice or a_cavity, b = b_lattice or b_cavity
        # only D(a_cavity, b_cavity)
        # are non-zero for cavity loss dissipator L="a". This can be shown analytically.

        dissipators_DM_p = {"decayup_c": (sigmas_c[0] + 1.j * sigmas_c[1]) / 2,
                            "decaydown_c": (sigmas_c[0] - 1.j * sigmas_c[1]) / 2,
                            "dephasing_c": sigmas_c[2]}

        for key, value in dissipators_DM_p.items():
            dissipators_POVM[key] = {'cc': get_D(value, jnp.conj(value).T, self.M_c, self.T_inv_cc)}

        dissipators_POVM["photonloss_c"] = {'cc': get_D(a_c, adag_c, self.M_c, self.T_inv_cc)}

        #### Lattice
        a_l = aOp(self.inputDimLattice)
        adag_l = adagOp(self.inputDimLattice)
        sigmas_l = get_paulis(self.inputDimLattice)
        dissipators_DM_l = {"decayup_l": (sigmas_l[0] + 1.j * sigmas_l[1]) / 2,
                            "decaydown_l": (sigmas_l[0] - 1.j * sigmas_l[1]) / 2,
                            "dephasing_l": sigmas_l[2]}

        for key, value in dissipators_DM_l.items():
            dissipators_POVM[key] = {'ll': get_D(value, jnp.conj(value).T, self.M_l, self.T_inv_ll)}

        dissipators_POVM["photonloss_l"] = {'ll': get_D(a_c, adag_c, self.M_c, self.T_inv_cc)}

        return dissipators_POVM

    def _get_unitaries(self):
        """Get 1- and 2-body unitary operators in the POVM formalism.

        Returns:
            Dictionary with common 1- and 2-body unitary operators.
        """
        unitaries_POVM = {}

        a_l = aOp(self.inputDimLattice)
        adag_l = adagOp(self.inputDimLattice)
        n_l = nOp(self.inputDimLattice)
        sigmas_l = get_paulis(self.inputDimLattice)

        a_c = aOp(self.inputDimCavity)
        adag_c = adagOp(self.inputDimCavity)
        n_c = nOp(self.inputDimCavity)
        sigmas_c = get_paulis(self.inputDimCavity)

        ######### 1-body operators ##########
        #### Lattice

        unitaries_DM_l = {"X_l": sigmas_l[0],
                          "Y_l": sigmas_l[1],
                          "Z_l": sigmas_l[2],
                          "a_l": a_l,
                          "adag_l": adag_l,
                          "n_l": n_l,
                          "n_l(n_l-1)": n_l @ n_l - n_l,
                          }
        for key, value in unitaries_DM_l.items():
            unitaries_POVM[key] = {'ll': get_U(value, self.M_l, self.T_inv_ll)}

        #### Cavity

        unitaries_DM_c = {"X_c": sigmas_c[0],
                          "Y_c": sigmas_c[1],
                          "Z_c": sigmas_c[2],
                          "a_c": a_c,
                          "adag_c": adag_c,
                          "n_c": n_c,
                          "n_c(n_c-1)": n_c @ n_c - n_c,
                          }
        for key, value in unitaries_DM_c.items():
            unitaries_POVM[key] = {'cc': get_U(value, self.M_c, self.T_inv_cc)}

        ######### 2-body operators ##########

        #### Involving only the lattice

        unitaries_DM_ll = {"X_l_X_l": jnp.kron(sigmas_l[0], sigmas_l[0]),
                           "Y_l_Y_l": jnp.kron(sigmas_l[1], sigmas_l[1]),
                           "Z_l_Z_l": jnp.kron(sigmas_l[2], sigmas_l[2]),
                           "a_l_adag_l": jnp.kron(a_l, adag_l),
                           "adag_l_a_l": jnp.kron(adag_l, a_l),
                           }

        for key, value in unitaries_DM_ll.items():
            unitaries_POVM[key] = {'llll': get_U(value, M_2Body(self.M_l, self.M_l), T_inv_2Body(self.T_inv_ll, self.T_inv_ll))}

        #### Interaction of the lattice and cavity
        unitaries_DM_lc = {"Sigma+_l_a_c": jnp.kron(sigmas_l[0] + 1.j * sigmas_l[1], a_c),
                           "Sigma-_l_adag_c": jnp.kron(sigmas_l[0] - 1.j * sigmas_l[1], adag_c),
                           "X_l_a_c": jnp.kron(sigmas_l[0], a_c),
                           "X_l_adag_c": jnp.kron(sigmas_l[0], adag_c),
                           "n_l_a_c": jnp.kron(n_l, a_c),
                           "n_l_adag_c": jnp.kron(n_l, adag_c),
                           "X_l_X_c": jnp.kron(sigmas_l[0], sigmas_c[0]),
                           "Y_l_Y_c": jnp.kron(sigmas_l[1], sigmas_c[1]),
                           "Z_l_Z_c": jnp.kron(sigmas_l[2], sigmas_c[2]),
                           }

        for key, value in unitaries_DM_lc.items():
            unitaries_POVM[key] = {'lclc': get_U(value, M_2Body(self.M_l, self.M_c), T_inv_2Body(self.T_inv_ll, self.T_inv_cc))}

        return unitaries_POVM

    def _get_observables(self):
        """Get X, Y, Z, n observables respectively for lattice and cavity in the POVM-formalism.

        Returns:
            Dictionary of dictionaries giving the X, Y, Z, n in the lattice and in the cavity.
        """
        observables_POVM = {}

        #### Lattice
        sigmas_l = get_paulis(self.inputDimLattice)
        n_l = nOp(self.inputDimLattice)
        observables_DM_l = {
            "X_l": sigmas_l[0],
            "Y_l": sigmas_l[1],
            "Z_l": sigmas_l[2],
            "n_l": n_l,
        }
        for key, value in observables_DM_l.items():
            observables_POVM[key] = {'lattice': get_Omega(value, self.M_l, self.T_inv_ll)}

        #### Cavity
        sigmas_c = get_paulis(self.inputDimCavity)
        n_c = nOp(self.inputDimCavity)
        observables_DM_c = {
            "X_c": sigmas_c[0],
            "Y_c": sigmas_c[1],
            "Z_c": sigmas_c[2],
            "n_c": n_c,
        }
        for key, value in observables_DM_c.items():
            observables_POVM[key] = {'cavity': get_Omega(value, self.M_c, self.T_inv_cc)}

        return observables_POVM


    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def _evaluate_observable(self, obs, idx):
        return obs[idx]

    def evaluate_observable(self, operator, states):
        """
        Obtain X, Y, Z, n and their correlators up to the specified length in ``SP_POVM.maxCorrLength``.

        Returns local magnetizations, and the number of particles on the lattice and in the cavity.
        """

        resPerSamplePerSpin = self._evaluate_magnetization_pmapd(operator, states)

        corrPerSamplePerSpin = {}
        # TODO
        # for corrLen in np.arange(self.maxCorrLength):
        #     corrPerSamplePerSpin[corrLen] = self._evaluate_correlators_pmapd(resPerSamplePerSpin, corrLen)

        return resPerSamplePerSpin, corrPerSamplePerSpin


class Operator(jVMC.operator.Operator):
    def __init__(self):
        super().__init__()
        self._alloc_Oloc_pmapd = global_defs.pmap_for_my_devices(
            lambda s: jnp.zeros(s.shape[0], dtype=global_defs.tReal))
        # in the POVM space, the values of Oloc are real
        # this is a hotfix for a bug with Oloc_batched in TDVP


class POVMOperator_LC(Operator):
    """This class provides functionality to compute operator matrix elements.

    Initializer arguments:

        * ``povm``: An instance of the POVM_LC-class.

    """

    def __init__(self, povm):
        """Initialize ``Operator``.
        """
        self.povm = povm
        self.ops = []
        super().__init__()

    def add(self, opDescr):
        """Add another operator to the operator.

        Args:
            * ``opDescr``: Operator dictionary to be added to the operator.
        """
        self.ops.append(opDescr)
        self.compiled = False


    def compile(self):
        self.ops_one_body_cc = [op for op in self.ops if 'cc' in self.povm.operators[op['name']] and len(op['sites']) == 1]
        self.ops_one_body_ll = [op for op in self.ops if 'll' in self.povm.operators[op['name']] and len(op['sites']) == 1]
        self.ops_two_body_llll = [op for op in self.ops if 'llll' in self.povm.operators[op['name']] and len(op['sites']) == 2]
        self.ops_two_body_lclc = [op for op in self.ops if 'lclc' in self.povm.operators[op['name']] and len(op['sites']) == 2]

        self.sp_site_idxs_ll = jnp.arange(self.povm.inputDimLattice ** 2)
        self.sp_site_idxs_cc = jnp.arange(self.povm.inputDimCavity ** 2)

        self.lDims_llll = jnp.array([self.povm.inputDimLattice ** 2, self.povm.inputDimLattice ** 2])
        self.lDims_lclc = jnp.array([self.povm.inputDimLattice ** 2, self.povm.inputDimCavity ** 2])


        self.sp_site_idxs_llll = jnp.array([[[i, j] for i in range(self.povm.inputDimLattice**2)] for j in range(self.povm.inputDimLattice**2)]).reshape((-1, 2))
        self.sp_site_idxs_lclc = jnp.array([[[i, j] for i in range(self.povm.inputDimLattice**2)] for j in range(self.povm.inputDimCavity**2)]).reshape((-1, 2))

        def _get_sites_matrices(op_list, subspace):  # take together operators acting on the same site
            sites_matrices = {}
            for op in op_list:

                if op['sites'] in sites_matrices:
                    sites_matrices[op['sites']] = sites_matrices[op['sites']] + self.povm.operators[op['name']][subspace] * op['strength']
                else:
                    sites_matrices[op['sites']] = self.povm.operators[op['name']][subspace] * op['strength']
            keys = [list(x) for x in sites_matrices.keys()]
            vals = list(sites_matrices.values())
            return jnp.array(keys), jnp.array(vals)
            # sites = jnp.array([op['sites'] for op in op_list])
            # matrices = jnp.array([self.povm.operators[op['name']][subspace]*op['strength'] for op in op_list])
            # return sites, matrices

        self.sites_cc, self.matrices_cc = _get_sites_matrices(self.ops_one_body_cc, 'cc')
        self.sites_ll, self.matrices_ll = _get_sites_matrices(self.ops_one_body_ll, 'll')
        self.sites_llll, self.matrices_llll = _get_sites_matrices(self.ops_two_body_llll, 'llll')
        self.sites_lclc, self.matrices_lclc = _get_sites_matrices(self.ops_two_body_lclc, 'lclc')

        return functools.partial(_get_s_primes,
                                 sites_cc=self.sites_cc, matrices_cc=self.matrices_cc,
                                 sp_site_idxs_cc=self.sp_site_idxs_cc,
                                 sites_ll=self.sites_ll, matrices_ll=self.matrices_ll,
                                 sp_site_idxs_ll=self.sp_site_idxs_ll,
                                 sites_llll=self.sites_llll, matrices_llll=self.matrices_llll,
                                 sp_site_idxs_llll=self.sp_site_idxs_llll, lDims_llll=self.lDims_llll,
                                 sites_lclc=self.sites_lclc, matrices_lclc=self.matrices_lclc,
                                 sp_site_idxs_lclc=self.sp_site_idxs_lclc, lDims_lclc=self.lDims_lclc)



def _get_s_primes(s, *args,
                  sites_cc, matrices_cc, sp_site_idxs_cc,
                  sites_ll, matrices_ll, sp_site_idxs_ll,
                  sites_llll, matrices_llll, sp_site_idxs_llll, lDims_llll,
                  sites_lclc, matrices_lclc, sp_site_idxs_lclc, lDims_lclc):

    def _update_one_site(s, site_idx, sp_site_idxs):
        return jax.vmap(lambda sp_site_idx: s.at[site_idx].set(sp_site_idx))(sp_site_idxs)

    def _matEl_one_site(op_matrix, s, site_idx, sp_site_idxs):
        return jnp.squeeze(jax.vmap(lambda sp_site_idx: op_matrix[s[site_idx], sp_site_idx])(sp_site_idxs), axis=-1)

    def _update_two_sites(s, site_idx, sp_site_idxs):
        return jax.vmap(lambda sp_site_idx: s.at[site_idx[0]].set(sp_site_idx[0]).at[site_idx[1]].set(sp_site_idx[1]))(sp_site_idxs)

    def _matEl_two_sites(op_matrix, s, site_idx, sp_site_idxs, lDims):
        return jax.vmap(lambda sp_site_idx: op_matrix[s[site_idx[0]] * lDims[1] + s[site_idx[1]], sp_site_idx[0] * lDims[1] + sp_site_idx[1]], in_axes=(0,))(sp_site_idxs)

    def _squeeze(arr):
        return arr.reshape((-1, arr.shape[-1]))

    sp_cc = _squeeze(jax.vmap(_update_one_site, in_axes=(None, 0, None))(s, sites_cc, sp_site_idxs_cc))
    matEl_cc = jax.vmap(_matEl_one_site, in_axes=(0, None, 0, None))(matrices_cc, s, sites_cc, sp_site_idxs_cc).ravel()

    sp_ll = _squeeze(jax.vmap(_update_one_site, in_axes=(None, 0, None))(s, sites_ll, sp_site_idxs_ll))
    matEl_ll = jax.vmap(_matEl_one_site, in_axes=(0, None, 0, None))(matrices_ll, s, sites_ll, sp_site_idxs_ll).ravel()

    sp_llll = _squeeze(jax.vmap(_update_two_sites, in_axes=(None, 0, None))(s, sites_llll, sp_site_idxs_llll))
    matEl_llll = jax.vmap(_matEl_two_sites, in_axes=(0, None, 0, None, None))(matrices_llll, s, sites_llll, sp_site_idxs_llll, lDims_llll).ravel()

    sp_lclc = _squeeze(jax.vmap(_update_two_sites, in_axes=(None, 0, None))(s, sites_lclc, sp_site_idxs_lclc))
    matEl_lclc = jax.vmap(_matEl_two_sites, in_axes=(0, None, 0, None, None))(matrices_lclc, s, sites_lclc, sp_site_idxs_lclc, lDims_lclc).ravel()


    sp = jnp.concatenate((sp_cc, sp_ll, sp_llll, sp_lclc), axis=0)
    matEl = jnp.concatenate((matEl_cc, matEl_ll, matEl_llll, matEl_lclc), axis=0)

    # sp may contain the same state many times but this is not a problem (?)

    return sp, matEl
