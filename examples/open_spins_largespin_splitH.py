from jax.config import config
import jax.numpy as jnp
import flax.linen as nn

import matplotlib.pyplot as plt
from functools import partial
import time
import numpy as np
import jax.random as random
import copy

import qutip

import jVMC
from jVMC_cavity.povm import POVM_LC, POVMOperator_LC, measure_povm_LC
from jVMC_cavity.sampler import ExactSampler_LC
from jVMC_cavity.symmetries import get_orbit_1d_LC
from jVMC_cavity.nets.rnn1d_general import RNN1DGeneral_LC2Sym
from jVMC_cavity.utils import normalize, norm_fun, set_initial_state
from jVMC_cavity.tdvp_batched import TDVP_batched

config.update("jax_enable_x64", True)


def to_qutip_state(state_vector, inputDim):
    psi = state_vector[0] * qutip.basis(inputDim, 0)
    for i in range(1, inputDim):
        psi += state_vector[i] * qutip.basis(inputDim, i)
    return psi

### PHYSICAL SYSTEM ###
L = 4
inputDimLattice = 2
inputDimCavity = 3


w_l = 1.0  # lattice spin z magnetic field
w_c = 1.0  # cavity largespin z magnetic field
g = 1.0  # spin-largespin ZZ coupling strength
delta_l = 1.0  # lattice spin-spin ZZ coupling strength
b_x_l = 1.0  # x magnetic field spin lattice
b_x_c = 1.0  # x magnetic field largespin cavity

kappa_c = 3.0  # cavity largespin decaydown rate
gamma_l = 3.0  # lattice spin decaydown rate
Gamma_l = 0.0  # lattice spin decayup rate

tmax = 50
dt = 1e-3


print(f"exact_dim={inputDimLattice**L * inputDimCavity}")

initial_lattice_state_vector = np.zeros(inputDimLattice)
initial_cavity_state_vector = np.zeros(inputDimCavity)

# define initial state
initial_lattice_state_vector[:] = 1.
initial_cavity_state_vector[:] = 1.

initial_lattice_state_vector = normalize(initial_lattice_state_vector)
initial_cavity_state_vector = normalize(initial_cavity_state_vector)

### EXACT CALCULATIONS ###
# initial state in the qutip basis
psi_spin = to_qutip_state(initial_lattice_state_vector, inputDimLattice)
psi_cavity = to_qutip_state(initial_cavity_state_vector, inputDimCavity)

psi_lattice = [psi_spin] * L
psi0 = qutip.tensor(*psi_lattice, psi_cavity)

# operators
S_l = (inputDimLattice-1.)/2.
X_l, Y_l, Z_l = qutip.operators.jmat(S_l)

qeye_lattice = [qutip.qeye(inputDimLattice)] * L


def op_lattice_tensor(op, idx):
    A = copy.deepcopy(qeye_lattice)
    A[idx] = op
    return qutip.tensor(*A, qutip.qeye(inputDimCavity))


sm_l = [op_lattice_tensor((X_l - 1.j * Y_l) / 2., i) for i in range(L)]
sx_l = [op_lattice_tensor(X_l, i) for i in range(L)]
sy_l = [op_lattice_tensor(Y_l, i) for i in range(L)]
sz_l = [op_lattice_tensor(Z_l, i) for i in range(L)]


S_c = (inputDimCavity-1.)/2.
X_c, Y_c, Z_c = qutip.operators.jmat(S_c)
sx_c = qutip.tensor(*qeye_lattice, X_c)
sy_c = qutip.tensor(*qeye_lattice, Y_c)
sz_c = qutip.tensor(*qeye_lattice, Z_c)
sm_c = qutip.tensor(*qeye_lattice, (X_c - 1.j * Y_c) / 2.)

# Hamiltonian
H = w_c * sz_c + b_x_c * sx_c

for i in range(L):
    H += w_l * sz_l[i]
    H += b_x_l * sx_l[i]
    H += delta_l * sz_l[i] * sz_l[(i + 1) % L]
    H += g * sz_l[i] * sz_c

# collapse operators
c_ops = [np.sqrt(kappa_c) * sm_c,  # cavity decaydown
         *[np.sqrt(gamma_l) * op for op in sm_l],  # lattice decaydown
         *[np.sqrt(Gamma_l) * op.dag() for op in sm_l],  # lattice decayup
         ]


# solve Lindblad equation and calculate observables
tlist = np.arange(0, tmax, dt)
opt = qutip.Odeoptions(nsteps=2000)  # allow extra time-steps
output_exact = qutip.mesolve(H, psi0, tlist, c_ops, [*sz_l, sz_c], options=opt)  # exact
# output_exact = qutip.mcsolve(H, psi0, tlist, c_ops, [*sz_l, sz_c], ntraj=500, options=opt)  # monte carlo

Z_l_exact = np.array(output_exact.expect[:L])
Z_c_exact = output_exact.expect[L]


def plot_exact():
    plt.plot(tlist, Z_c_exact, '--b', alpha=0.5, label="exact $Z_c$")
    plt.plot(tlist, Z_l_exact[0], '--r', alpha=0.5, label="exact $Z_l$")  # translational invariance, so we can plot only one Z_l[i]
    plt.xlabel('time')
    plt.ylabel('')


plt.ion()
plot_exact()
plt.legend()
plt.pause(1)
plt.clf()


### NQS + POVM CALCULATIONS ###

hiddenSizeLattice = 5
hiddenSizeCavity = 5
depth = 3
initScale = 1.0
cell = "RNN"
psi_kwargs = dict(batchSize=2000, seed=1234)
sampler_id = "a"  # "e" for exact sampling, "a" for autoregressive sampling
logProbFactor = 1
sampler_kwargs = dict(numSamples=4000)
icPOVMCavity = 'orthocross'  #'symmetric'
icPOVMLattice = 'orthocross'  #'symmetric'

exact_dim = inputDimLattice ** (2 * L) * inputDimCavity ** 2

povm = POVM_LC(L,
               inputDimCavity=inputDimCavity,
               inputDimLattice=inputDimLattice,
               maxCorrLength=0,
               icPOVMCavity=icPOVMCavity,
               icPOVMLattice=icPOVMLattice)

# Initialize net
sample_shape = (L + 1,)
orbit = get_orbit_1d_LC(L, translation=True, reflection=False)

net_kwargs = dict(L=L,
                  hiddenSizeLattice=hiddenSizeLattice,
                  hiddenSizeCavity=hiddenSizeCavity,
                  depth=depth,
                  inputDimLattice=inputDimLattice ** 2,  # for SIC-POVM
                  actFun=nn.elu,
                  inputDimCavity=inputDimCavity ** 2,  # for SIC-POVM
                  initScale=initScale,
                  logProbFactor=logProbFactor,
                  realValuedOutput=True,
                  realValuedParams=True,
                  cell=cell,
                  orbit=orbit,
                  )

net = RNN1DGeneral_LC2Sym(**net_kwargs)

psi = jVMC.vqs.NQS(net, **psi_kwargs)
# to compile one has to evaluate once with a certain dimensional input
psi(jnp.zeros((jVMC.global_defs.device_count(), 1) + sample_shape, dtype=jnp.int32))
print(f"The variational ansatz has {psi.numParameters} parameters.")

set_initial_state(psi, povm, initial_lattice_state_vector, subspace="lattice")
set_initial_state(psi, povm, initial_cavity_state_vector, subspace="cavity")

if sampler_id == "e":
    sampler = ExactSampler_LC(psi, sample_shape,
                              inputDimLattice=inputDimLattice ** 2,  # for SIC-POVM
                              inputDimCavity=inputDimCavity ** 2,  # for SIC-POVM
                              logProbFactor=logProbFactor)
elif sampler_id == "a":
    sampler = jVMC.sampler.MCSampler(psi, (L + 1,), random.PRNGKey(123), **sampler_kwargs)
else:
    raise ValueError(f"sampler_id should be one of 'a', 'e'.")

Lindbladian = POVMOperator_LC(povm)

# # Hamiltonian
Lindbladian.add({"name": "Z_c", "strength": w_c, "sites": (L,)})  # H = w_c * sz_c
Lindbladian.add({"name": "X_c", "strength": b_x_c, "sites": (L,)})  # H += b_x_c * sx_c

for i in range(L):
    Lindbladian.add({"name": "Z_l", "strength": w_l, "sites": (i,)})  # H += w_l * sz_l[i]
    Lindbladian.add({"name": "X_l", "strength": b_x_l, "sites": (i,)})  # H += b_x_l * sx_l[i]
    Lindbladian.add(
        {"name": "Z_l_Z_l", "strength": delta_l, "sites": (i, (i + 1) % L)})  # H += delta * sz_l[i] * sz_l[(i + 1) % L]
    Lindbladian.add({"name": "Z_l_Z_c", "strength": g, "sites": (i, L)})  # H += g * sz_l[i] * sz_c

# Dissipative part
Lindbladian.add({"name": "decaydown_c", "strength": kappa_c, "sites": (L,)})  # np.sqrt(kappa_c) * sm_c
for i in range(L):
    Lindbladian.add(
        {"name": "decaydown_l", "strength": gamma_l, "sites": (i,)})  # *[np.sqrt(gamma_l) * op for op in sm_l]
    Lindbladian.add(
        {"name": "decayup_l", "strength": Gamma_l, "sites": (i,)})  # *[np.sqrt(Gamma_l) * op.dag() for op in sm_l]

outp = jVMC.util.OutputManager("outp_open_spins_largespin.hdf5", append=False)

# Set up TDVP
# tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
#                                    svdTol=1e-8, diagonalShift=0, snrTol=2, diagonalizeOnDevice=False,
#                                    makeReal='real', crossValidation=False)
tdvpEquation = TDVP_batched(sampler, rhsPrefactor=-1.,
                                   svdTol=1e-6, diagonalShift=0, snrTol=2, diagonalizeOnDevice=False,
                                   makeReal='real', crossValidation=False)

# ODE integrator
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=dt, tol=1e-4)
# stepper = jVMC.util.stepper.Euler(timeStep=dt)

t = 0
times = []
data = []

while t < tmax:
    tic = time.perf_counter()
    times.append(t)
    result = measure_povm_LC(Lindbladian.povm, sampler)
    data.append([t, *(result['Z_l']['mean']), result['Z_c']['mean'][0]])

    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(),
                          hamiltonian=Lindbladian, psi=psi, outp=outp,
                          normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
    t += dt
    psi.set_parameters(dp)
    print(f"t = {t:.3f}, \t dt = {dt:.2e}")

    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))
    toc = time.perf_counter()
    print("   == Total time for this step: %fs\n" % (toc - tic))
    if tdvpEquation.crossValidation:
        print(f"Cross-Validation-Factor_residual = {tdvpEquation.crossValidationFactor_residual:.3f}")
        print(f"Cross-Validation-Factor_tdvpErr = {tdvpEquation.crossValidationFactor_tdvpErr:.3f}")

    outp.write_observables(t, **result)

    # Plot data
    npdata = np.array(data)

    plot_exact()
    plt.plot(npdata[:, 0], np.mean(npdata[:, 1:L + 1], axis=1), '-r', label=f'$Z_l$')
    plt.plot(npdata[:, 0], npdata[:, L + 1], '-b', label='$Z_c$')

    plt.legend()
    plt.pause(0.05)
    plt.clf()


plot_exact()
plt.plot(npdata[:, 0], np.mean(npdata[:, 1:L + 1], axis=1), '-r', label=f'$Z_l$')
plt.plot(npdata[:, 0], npdata[:, L + 1], '-b', label='$Z_c$')

plt.legend()
plt.pause(100000000)
