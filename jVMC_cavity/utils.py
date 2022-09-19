import jax.numpy as jnp
import numpy as np
import jax

from .povm import get_1_particle_distributions_LC, POVM_LC
from jVMC.vqs import NQS


def normalize(v):
    if isinstance(v, jnp.ndarray):
        norm = jnp.linalg.norm(v)
    else:
        norm = np.linalg.norm(v)

    if norm == 0:
            return v
    return v / norm


def copy_dict(a):
    b = {}
    for key, value in a.items():
        if type(value) == type(a):
            b[key] = copy_dict(value)
        else:
            b[key] = value
    return b


def norm_fun(v, df=lambda x: x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))


def set_initial_state(psi: NQS, povm: POVM_LC, state_vector: np.ndarray, subspace: str):
    """ Sets initial state on NQS psi by setting the biases of the last layer and kernel of the
    last layer to 1e-15 * kernel.
    """
    # Initial state on NQS in the POVM language
    params = copy_dict(psi._param_unflatten(psi.get_parameters()))

    outDense = ""
    if subspace == "lattice":
        outDense = "outputDenseLattice"
    elif subspace == "cavity":
        outDense = "outputDenseCavity"
    else:
        raise ValueError("subspace must be one of the 'lattice' or 'cavity'.")

    prob_dist_subspace = get_1_particle_distributions_LC(state_vector, povm, subspace=subspace)
    biases_subspace = jnp.log(prob_dist_subspace)

    def set_bias_kernel(params):
        for k, v in params.items():
            if k == outDense:
                v["last_layer"]["bias"] = biases_subspace
                v["last_layer"]["kernel"] = 1e-15 * v["last_layer"]["kernel"]
                pass
            elif isinstance(v, dict):
                set_bias_kernel(v)

    set_bias_kernel(params)
    params = jnp.concatenate([p.ravel()
                              for p in jax.tree_util.tree_flatten(params)[0]])
    psi.set_parameters(params)
    pass
