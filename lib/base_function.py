import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time
# import jax.numpy as jnp
import scipy.integrate as integrate
import concurrent.futures as cf


def get_coupling_parameter(
    Kc: np.float64,
    N: np.int64
) -> npt.NDArray[np.float64]:

    plus = Kc + 0.001 * \
        np.logspace(start=1, stop=np.log2(4*Kc*1000),
                    num=N, endpoint=False, base=2)
    minus = Kc - 0.001 * \
        np.logspace(start=1, stop=np.log2(Kc*1000),
                    num=N, endpoint=False, base=2)
    K = np.concatenate([np.array([Kc]), plus, minus])

    return np.array(sorted(K))


def exact_free_energy(
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    free_energy = np.zeros(len(beta))

    for i, K in enumerate(beta):
        k = 1 / (np.sinh(2*K)**2)

        def integrand(theta): return np.log((np.cosh(2*K))**2
                                            + 1 / k * np.sqrt(1 + k**2 - 2*k * np.cos(2*theta)))

        free_energy[i] = np.log(2) / 2 + 1 / (2 * np.pi) * \
            integrate.quad(integrand, 0, np.pi)[0]

    return free_energy


def exact_order_parameter(
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    order_parameter = np.zeros(len(beta))
    Kc = np.log(1+np.sqrt(2))/2

    for i, K in enumerate(beta):
        if (K <= Kc):
            order_parameter[i] = 0

        elif (K > Kc):
            order_parameter[i] = (1-np.sinh(2*K)**(-4))**(1/8)

    return order_parameter


def exact_internal_energy(
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    free_energy = exact_free_energy(beta)

    energy = -np.gradient(free_energy) * 1 / (beta[1] - beta[0])

    return energy


def exact_heat_capacity(
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    energy = exact_internal_energy(beta)

    heat_capacity = -np.gradient(energy) * 1 / (beta[1]-beta[0]) * beta**2

    return heat_capacity


def contract_four(
    T1: npt.NDArray[np.float64],
    T2: npt.NDArray[np.float64],
    T3: npt.NDArray[np.float64],
    T4: npt.NDArray[np.float64],
    mode: str,
) -> tuple[str, npt.NDArray[np.float64]]:

    return mode, np.log(np.abs(
        np.einsum("abcd,efgb,ghij,cjkl->", T1, T2, T3, T4, optimize=True)))
