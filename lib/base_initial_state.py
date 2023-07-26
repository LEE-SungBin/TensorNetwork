import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time

from lib.base_dataclass import Base_Input


def Bessel(
    n: npt.NDArray[np.int64],
    beta: float
) -> npt.NDArray[np.float64]:

    return sp.special.iv(n, beta)


def get_pure_state(
    input: Base_Input
) -> npt.NDArray[np.float64]:

    state, beta, magnetic_field = (
        input.parameter.state,
        input.parameter.beta,
        input.parameter.magnetic_field,
    )

    if state == 0:
        return get_XY_state(input)

    # * Ising model = state 2 potts model

    else:
        Z_nn_interaction = np.zeros([state, state], dtype=np.float64)
        theta = np.array([i/state*2*np.pi for i in range(state)])
        inds = np.arange(state)  # * [0, 1]
        for i, j in itertools.product(inds, inds):
            Z_nn_interaction[i][j] = np.exp(beta*np.cos(theta[i]-theta[j]))

        Z_external_field = np.array(
            [np.exp(beta*magnetic_field*np.cos(angle)) for angle in theta])

        # * eigen-decompostion
        Z_nn_eigenvalue, Z_nn_eigenvector = np.linalg.eigh(Z_nn_interaction)
        Z_nn_eigenvalue, Z_nn_eigenvector = np.diag(
            np.flip(Z_nn_eigenvalue, axis=(0,))), np.flip(Z_nn_eigenvector, axis=(1,))

        Z_nn_interaction_U, Z_nn_interaction_UT = Z_nn_eigenvector @ np.sqrt(
            Z_nn_eigenvalue), (Z_nn_eigenvector @ np.sqrt(Z_nn_eigenvalue)).T

        # theory = np.array([
        #     [np.sqrt(np.cosh(beta)), -np.sqrt(np.sinh(beta))],
        #     [np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))]
        # ])
        # print(np.allclose(theory, Z_nn_interaction_U))

        lattice = np.einsum(
            "ai,aj,ak,al,a->ijkl", Z_nn_interaction_U, Z_nn_interaction_U,
            Z_nn_interaction_U, Z_nn_interaction_U, Z_external_field)

        return lattice


def get_higher_order_moment(
    input: Base_Input,
    order: int,
) -> npt.NDArray[Any]:

    state, beta, magnetic_field = (
        input.parameter.state,
        input.parameter.beta,
        input.parameter.magnetic_field,
    )

    if state == 0:
        return get_higher_order_XY(input, order)

    else:
        Z_nn_interaction = np.zeros([state, state], dtype=np.float64)
        theta = np.array([i/state*2*np.pi for i in range(state)])
        inds = np.arange(state)
        for i, j in itertools.product(inds, inds):
            Z_nn_interaction[i][j] = np.exp(beta*np.cos(theta[i]-theta[j]))

        Z_external_field = np.array(
            [np.exp(beta*magnetic_field*np.cos(angle)) for angle in theta])

        Z_nn_eigenvalue, Z_nn_eigenvector = np.linalg.eigh(Z_nn_interaction)
        Z_nn_eigenvalue, Z_nn_eigenvector = np.diag(
            np.flip(Z_nn_eigenvalue, axis=(0,))), np.flip(Z_nn_eigenvector, axis=(1,))

        Z_nn_interaction_U, Z_nn_interaction_UT = Z_nn_eigenvector @ np.sqrt(
            Z_nn_eigenvalue), (Z_nn_eigenvector @ np.sqrt(Z_nn_eigenvalue)).T

        magnetization = np.array([np.exp(angle*1j) for angle in theta])

        lattice = np.einsum(
            "ai,aj,ak,al,a,a->ijkl", Z_nn_interaction_U, Z_nn_interaction_U,
            Z_nn_interaction_U, Z_nn_interaction_U, Z_external_field,
            magnetization**order, optimize=True)

        return lattice


def get_XY_state(
    input: Base_Input
) -> npt.NDArray[np.float64]:

    beta, magnetic_field, Dcut = (
        input.parameter.beta,
        input.parameter.magnetic_field,
        input.RG_operation.Dcut
    )

    index = np.arange(Dcut)

    lattice = np.zeros((Dcut, Dcut, Dcut, Dcut), dtype=np.float64)

    for i, j, k, l in itertools.product(index, index, index, index):
        lattice[i][j][k][l] = np.sqrt(
            Bessel(i, beta) * Bessel(j, beta) *
            Bessel(k, beta) * Bessel(l, beta)
        ) * Bessel(i+j-k-l, beta*magnetic_field)

    return lattice


def get_higher_order_XY(
    input: Base_Input,
    order: int
) -> npt.NDArray[np.float64]:

    beta, magnetic_field, Dcut = (
        input.parameter.beta,
        input.parameter.magnetic_field,
        input.RG_operation.Dcut
    )

    index = np.arange(Dcut)

    lattice = np.zeros((Dcut, Dcut, Dcut, Dcut), dtype=np.float64)

    for i, j, k, l in itertools.product(index, index, index, index):
        lattice[i][j][k][l] = np.sqrt(
            Bessel(i, beta) * Bessel(j, beta) *
            Bessel(k, beta) * Bessel(l, beta)
        ) * (Bessel(i+j-k-l-order, beta*magnetic_field) +
             Bessel(i+j-k-l+order, beta*magnetic_field)) / 2.0

    return lattice
