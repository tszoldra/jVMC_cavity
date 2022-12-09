import jax.numpy as jnp
import jVMC.mpi_wrapper as mpi
from jVMC.util.tdvp import TDVP


class TDVP_batched(TDVP):
    """ This class provides functionality to solve a time-dependent variational principle (TDVP).

    With the force vector

        :math:`F_k=\langle \mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c`

    and the quantum Fisher matrix

        :math:`S_{k,k'} = \langle \mathcal O_{\\theta_k} (\mathcal O_{\\theta_{k'}})^*\\rangle_c`

    and for real parameters :math:`\\theta\in\mathbb R`, the TDVP equation reads

        :math:`q\\big[S_{k,k'}\\big]\\theta_{k'} = -q\\big[xF_k\\big]`

    Here, either :math:`q=\\text{Re}` or :math:`q=\\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    For ground state search a regularization controlled by a parameter :math:`\\rho` can be included
    by increasing the diagonal entries and solving

        :math:`q\\big[(1+\\rho\delta_{k,k'})S_{k,k'}\\big]\\theta_{k'} = -q\\big[F_k\\big]`

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

        :math:`S = V\Sigma V^\dagger`

    with a diagonal matrix :math:`\Sigma_{kk}=\sigma_k`
    Assuming that :math:`\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed 
    from the regularized inverted eigenvalues

        :math:`\\tilde\sigma_k^{-1}=\\frac{1}{\\Big(1+\\big(\\frac{\epsilon_{SVD}}{\sigma_j/\sigma_1}\\big)^6\\Big)\\Big(1+\\big(\\frac{\epsilon_{SNR}}{\\text{SNR}(\\rho_k)}\\big)^6\\Big)}`

    with :math:`\\text{SNR}(\\rho_k)` the signal-to-noise ratio of :math:`\\rho_k=V_{k,k'}^{\dagger}F_{k'}` (see `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``snrTol``: Regularization parameter :math:`\epsilon_{SNR}`, see above.
        * ``svdTol``: Regularization parameter :math:`\epsilon_{SVD}`, see above.
        * ``makeReal``: Specifies the function :math:`q`, either `'real'` for :math:`q=\\text{Re}` or `'imag'` for :math:`q=\\text{Im}`.
        * ``rhsPrefactor``: Prefactor :math:`x` of the right hand side, see above.
        * ``diagonalShift``: Regularization parameter :math:`\\rho` for ground state search, see above.
        * ``crossValidation``: Perform cross-validation check as introduced in `[arXiv:2105.01054] <https://arxiv.org/pdf/2105.01054.pdf>`_.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, netParameters, t, *, psi, hamiltonian, **rhsArgs):
        """ For given network parameters this function solves the TDVP equation calculating Eloc in batches.

        This function returns :math:`\\dot\\theta=S^{-1}F`. Thereby an instance of the ``TDVP`` class is a suited
        callable for the right hand side of an ODE to be used in combination with the integration schemes 
        implemented in ``jVMC.stepper``. Alternatively, the interface matches the scipy ODE solvers as well.

        Avoids the OOM issues for operators with a large number of offdiagonal terms by batching,
        with a batch size specified by ``psi.batchSize``.
        It makes sense to use it if the number of offdiagonal terms for all operators
        is larger than the number of parameters in the NQS.

        Arguments:
            * ``netParameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``psi``: NQS ansatz. Instance of ``jVMC.vqs.NQS``.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements. 

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.

        Returns:
            The solution of the TDVP equation, :math:`\\dot\\theta=S^{-1}F`.
        """

        tmpParameters = psi.get_parameters()
        psi.set_parameters(netParameters)

        outp = None
        if "outp" in rhsArgs:
            outp = rhsArgs["outp"]
        self.outp = outp

        numSamples = None
        if "numSamples" in rhsArgs:
            numSamples = rhsArgs["numSamples"]

        def start_timing(outp, name):
            if outp is not None:
                outp.start_timing(name)

        def stop_timing(outp, name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if outp is not None:
                outp.stop_timing(name)

        # Get sample
        start_timing(outp, "sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=numSamples)
        stop_timing(outp, "sampling", waitFor=sampleConfigs)

        # # Evaluate local energy
        # start_timing(outp, "compute Eloc")
        # sampleOffdConfigs, matEls = hamiltonian.get_s_primes(sampleConfigs, t)
        # start_timing(outp, "evaluate off-diagonal")
        # sampleLogPsiOffd = psi(sampleOffdConfigs)
        # stop_timing(outp, "evaluate off-diagonal", waitFor=sampleLogPsiOffd)
        # Eloc = hamiltonian.get_O_loc(sampleLogPsi, sampleLogPsiOffd)
        # stop_timing(outp, "compute Eloc", waitFor=Eloc)

        start_timing(outp, "compute Eloc")
        Eloc = hamiltonian.get_O_loc_batched(sampleConfigs, psi, sampleLogPsi, psi.batchSize)
        stop_timing(outp, "compute Eloc", waitFor=Eloc)

        # Evaluate gradients
        start_timing(outp, "compute gradients")
        sampleGradients = psi.gradients(sampleConfigs)
        stop_timing(outp, "compute gradients", waitFor=sampleGradients)

        start_timing(outp, "solve TDVP eqn.")
        update, solverResidual = self.solve(Eloc, sampleGradients, p)
        stop_timing(outp, "solve TDVP eqn.")

        if outp is not None:
            outp.add_timing("MPI communication", mpi.get_communication_time())

        psi.set_parameters(tmpParameters)

        if "intStep" in rhsArgs:
            if rhsArgs["intStep"] == 0:

                self.ElocMean0 = self.ElocMean
                self.ElocVar0 = self.ElocVar
                self.tdvpError = self._get_tdvp_error(update)
                self.solverResidual = solverResidual
                self.snr0 = self.snr
                self.ev0 = self.ev

                if self.crossValidation:

                    if p != None:
                        update_1, _ = self.solve(Eloc[:, 0::2], sampleGradients[:, 0::2], p[:, 0::2])
                        S2, F2, _ = self.get_tdvp_equation(Eloc[:, 1::2], sampleGradients[:, 1::2], p[:, 1::2])
                    else:
                        update_1, _ = self.solve(Eloc[:, 0::2], sampleGradients[:, 0::2])
                        S2, F2, _ = self.get_tdvp_equation(Eloc[:, 1::2], sampleGradients[:, 1::2])

                    validation_tdvpErr = self._get_tdvp_error(update_1)
                    update, solverResidual = self.solve(Eloc, sampleGradients, p)
                    validation_residual = (jnp.linalg.norm(S2.dot(update_1) - F2) / jnp.linalg.norm(F2)) / solverResidual

                    self.crossValidationFactor_residual = validation_residual
                    self.crossValidationFactor_tdvpErr = validation_tdvpErr / self.tdvpError

                    self.S, _, _ = self.get_tdvp_equation(Eloc, sampleGradients, p)

        return update
