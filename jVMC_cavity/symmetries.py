import numpy as np
import jax.numpy as jnp
import jax
from jVMC.util.symmetries import LatticeSymmetry
import flax.linen as nn


def get_orbit_1d_LC(L, translation=True, reflection=True, translation_by_2=False, **kwargs):
    """ This function generates the group of lattice symmetries in a one-dimensional lattice in a cavity.
    There are no symmetry operations applied to the cavity.

    Arguments:
        * ``L``: Linear dimension of the lattice.
        * ``reflection``: Boolean to indicate whether reflections are to be included
        * ``translation``: Boolean to indicate whether translations are to be included
        * ``translation_by_2``: Boolean to indicate whether translations by 2 are to be included

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        symmetry operations and the following two dimensions correspond to the corresponding permutation matrix.
    """
    if translation_by_2 and L % 2 != 0:
        raise ValueError('For translation_by_2 symmetry orbit, the lattice size must be an even number. PBC must be assumed.')


    def get_point_orbit_1D(L, reflection):
        if reflection:
            reflection_op = jnp.block([
                [jnp.fliplr(jnp.eye(L)), jnp.zeros((L, 1))],
                [jnp.zeros((1, L)), np.eye(1)]
            ])
            return jnp.array([jnp.eye(L + 1), reflection_op])
        else:
            return jnp.array([jnp.eye(L + 1)])

    def get_translation_orbit_1D(L, translation):
        to_s = np.array([np.eye(L)] * L)
        to_sp = [None] * L
        for idx, t in enumerate(to_s):
            to_s[idx] = np.roll(t, idx, axis=1)
            to_sp[idx] = np.block([
                [to_s[idx], np.zeros((L, 1))],
                [np.zeros((1, L)), np.eye(1)]
                ])
        if translation:
            return jnp.array(to_sp)
        else:
            return jnp.array([jnp.eye(L+1)])

    def get_translation_by_2_orbit_1D(L, translation_by_2):
        to_s = np.array([np.eye(L)] * (L // 2))
        to_sp = [None] * (L // 2)
        for idx, t in enumerate(to_s):
            to_s[idx] = np.roll(t, 2 * idx, axis=1)
            to_sp[idx] = np.block([
                [to_s[idx], np.zeros((L, 1))],
                [np.zeros((1, L)), np.eye(1)]
            ])
        if translation_by_2 and L >= 4:
            return jnp.array(to_sp)
        else:
            return jnp.array([jnp.eye(L + 1)])

    po = get_point_orbit_1D(L, reflection)
    to = get_translation_orbit_1D(L, translation)
    to_by_2 = get_translation_by_2_orbit_1D(L, translation_by_2)

    #orbit = jax.vmap(lambda x, y: jax.vmap(lambda a, b: jnp.dot(a, b), in_axes=(None, 0))(x, y, z), in_axes=(0, None))(to,
    #                                                                                                                po)
    orbit = jnp.einsum('aij, bjk, ckl -> abcil', to, po, to_by_2)

    orbit = orbit.reshape((-1, L+1, L+1))
    return LatticeSymmetry(orbit.astype(np.int32))