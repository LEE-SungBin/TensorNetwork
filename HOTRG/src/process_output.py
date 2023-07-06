import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time
# import jax.numpy as jnp
# from opt_einsum import contract
import concurrent.futures as cf

from HOTRG.src.dataclass import Input, Single_Result


def get_result(
    input: Input,
    pure: npt.NDArray[np.float64],
    first_order: npt.NDArray[np.complex128],
    pure_norms_UD: list,
    pure_norms_LR: list,
    first_order_norms_UD: list,
    first_order_norms_LR: list,
    current_step: int,
    SIZE: int,
) -> Single_Result:

    single_result = get_free_energy(
        pure, pure_norms_UD, pure_norms_LR, current_step, SIZE)

    order_parameter = get_order_parameter(
        pure, first_order,
        pure_norms_UD, pure_norms_LR,
        first_order_norms_UD, first_order_norms_LR,
        current_step, SIZE
    )

    single_result.order_parameter = order_parameter

    return single_result


def get_free_energy(
    pure: npt.NDArray[np.float64],
    pure_norms_UD: list,
    pure_norms_LR: list,
    current_step: int,
    SIZE: int
) -> Single_Result:

    free_energy = np.log(np.abs(
        np.einsum("ijij->", pure, optimize=True))) / SIZE

    for j, pure_UD, pure_LR in zip(
            np.arange(current_step+1),
            pure_norms_UD, pure_norms_LR
    ):
        remain = current_step-j
        free_energy += 2**(2*remain+1)*np.log(pure_UD) / SIZE
        free_energy += 2**(2*remain)*np.log(pure_LR) / SIZE

    return Single_Result(
        free_energy=free_energy,
        order_parameter=0.0,
        hamiltonian=0.0,
        heat_capacity=0.0,
    )


def get_order_parameter(
    pure: npt.NDArray[np.float64],
    first_order: npt.NDArray[np.complex128],
    pure_norms_UD: list,
    pure_norms_LR: list,
    first_order_norms_UD: list,
    first_order_norms_LR: list,
    current_step: int,
    SIZE: int,
) -> float:

    order_parameter = np.einsum("ijij->", first_order, optimize=True) / \
        np.einsum("ijij->", pure, optimize=True)

    for j, pure_UD, pure_LR, first_order_UD, first_order_LR in zip(
            np.arange(current_step+1),
            pure_norms_UD, pure_norms_LR,
            first_order_norms_UD, first_order_norms_LR
    ):
        remain = current_step-j
        order_parameter *= first_order_UD / pure_UD
        order_parameter *= first_order_LR / pure_LR

    return np.real(order_parameter)
