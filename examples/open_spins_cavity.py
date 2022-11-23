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
from jVMC_cavity.nets.rnn1d_general import RNN1DGeneral_LCSym
from jVMC_cavity.utils import normalize, norm_fun, set_initial_state

config.update("jax_enable_x64", True)


def to_qutip_state(state_vector, inputDim):
    psi = state_vector[0] * qutip.basis(inputDim, 0)
    for i in range(1, inputDim):
        psi += state_vector[i] * qutip.basis(inputDim, i)
    return psi

### PHYSICAL SYSTEM ###
L = 3
inputDimLattice = 2
inputDimCavity = 3


w_l = 1.0  # lattice spin z magnetic field
w_c = 1.0  # cavity frequency
g = 1.0  # spin-cavity coupling strength
delta_l = 1.0  # lattice spin-spin ZZ coupling strength
b_x_l = 1.0  # x magnetic field spin lattice

kappa_c = 2.0  # cavity photon loss rate
gamma_l = 1.0  # lattice spin decaydown rate
Gamma_l = 0.5  # lattice spin decayup rate

tmax = 3
dt = 1e-3


print(f"exact_dim={inputDimLattice**L * inputDimCavity}")

initial_lattice_state_vector = np.zeros(inputDimLattice)
initial_cavity_state_vector = np.zeros(inputDimCavity)

# define initial state
initial_lattice_state_vector[:] = 1.
initial_cavity_state_vector[0] = 1.

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


a_c = qutip.tensor(*qeye_lattice, qutip.destroy(inputDimCavity))

# Hamiltonian
H = w_c * a_c.dag() * a_c

for i in range(L):
    H += w_l * sz_l[i]
    H += b_x_l * sx_l[i]
    H += delta_l * sz_l[i] * sz_l[(i + 1) % L]
    H += g * sx_l[i] * (a_c.dag() + a_c)

# collapse operators
c_ops = [np.sqrt(kappa_c) * a_c,  # cavity photon loss
         *[np.sqrt(gamma_l) * op for op in sm_l],  # lattice decaydown
         *[np.sqrt(Gamma_l) * op.dag() for op in sm_l],  # lattice decayup
         ]


# solve Lindblad equation and calculate observables
tlist = np.arange(0, tmax, dt)
opt = qutip.Odeoptions(nsteps=2000)  # allow extra time-steps
output_exact = qutip.mesolve(H, psi0, tlist, c_ops, [*sz_l, a_c.dag() * a_c], options=opt)  # exact
# output_exact = qutip.mcsolve(H, psi0, tlist, c_ops, [*sz_l, a_c.dag() * a_c], ntraj=500, options=opt)  # monte carlo

Z_l_exact = np.array(output_exact.expect[:L])
n_c_exact = output_exact.expect[L]


def plot_exact():
    plt.plot(tlist, n_c_exact, '--b', alpha= 0.5, label="exact $n_c$")
    plt.plot(tlist, Z_l_exact[0], '--r', alpha=0.5, label="exact $Z_l$")  # translational invariance, so we can plot only one Z_l[i]
    plt.xlabel('time')
    plt.ylabel('observable value')

plt.ion()
plot_exact()
plt.legend()
plt.pause(1)
plt.clf()


### NQS + POVM CALCULATIONS ###

hiddenSize = 6
depth = 2
initScale = 1.0
cell = "LSTM"
denseCavityLayers = (9, 9,)
psi_kwargs = dict(batchSize=1000, seed=1234)
sampler_id = "e"  # "e" for exact sampling, "a" for autoregressive sampling
logProbFactor = 1
sampler_kwargs = dict(numSamples=1000)
icPOVMCavity = 'orthocross'#'symmetric'  # 'orthocross'
icPOVMLattice = 'orthocross' #'symmetric'  # 'orthocross'

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
                  hiddenSize=hiddenSize,
                  depth=depth,
                  inputDimLattice=inputDimLattice**2,  # for SIC-POVM
                  actFun=nn.elu,
                  inputDimCavity=inputDimCavity**2,  # for SIC-POVM
                  denseCavityLayers=denseCavityLayers,
                  initScale=initScale,
                  logProbFactor=logProbFactor,
                  realValuedOutput=True,
                  realValuedParams=True,
                  cell=cell,
                  orbit=orbit)

net = RNN1DGeneral_LCSym(**net_kwargs)

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

# Hamiltonian
H = w_c * a_c.dag() * a_c

for i in range(L):
    H += w_l * sz_l[i]
    H += b_x_l * sx_l[i]
    H += delta_l * sz_l[i] * sz_l[(i + 1) % L]
    H += g * sx_l[i] * (a_c.dag() + a_c)


# # Hamiltonian
Lindbladian.add({"name": "n_c", "strength": w_c, "sites": (L,)})  # H = w_c * sz_c

for i in range(L):
    Lindbladian.add({"name": "Z_l", "strength": w_l, "sites": (i,)})  # H += w_l * sz_l[i]
    Lindbladian.add({"name": "X_l", "strength": b_x_l, "sites": (i,)})  # H += b_x_l * sx_l[i]
    Lindbladian.add(
        {"name": "Z_l_Z_l", "strength": delta_l, "sites": (i, (i + 1) % L)})  # H += delta * sz_l[i] * sz_l[(i + 1) % L]
    Lindbladian.add({"name": "X_l_a_c", "strength": g, "sites": (i, L)})  # H += g * sx_l[i] * a_c
    Lindbladian.add({"name": "X_l_adag_c", "strength": g, "sites": (i, L)})  # H += g * sx_l[i] * a_c.dag()


# Dissipative part
Lindbladian.add({"name": "photonloss_c", "strength": kappa_c, "sites": (L,)})  # np.sqrt(kappa_c) * a_c
for i in range(L):
    Lindbladian.add(
        {"name": "decaydown_l", "strength": gamma_l, "sites": (i,)})  # *[np.sqrt(gamma_l) * op for op in sm_l]
    Lindbladian.add(
        {"name": "decayup_l", "strength": Gamma_l, "sites": (i,)})  # *[np.sqrt(Gamma_l) * op.dag() for op in sm_l]

outp = jVMC.util.OutputManager("outp_open_spins_cavity.hdf5", append=False)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
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
    data.append([t, *(result['Z_l']['mean']), result['n_c']['mean'][0]])

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
    plt.plot(npdata[:, 0], npdata[:, L + 1], '-b', label='$n_c$')

    plt.legend()
    plt.pause(0.05)
    plt.clf()

plot_exact()
plt.plot(npdata[:, 0], np.mean(npdata[:, 1:L + 1], axis=1), '-r', label=f'$Z_l$')
plt.plot(npdata[:, 0], npdata[:, L + 1], '-b', label='$n_c$')

plt.legend()
plt.pause(100000000)
