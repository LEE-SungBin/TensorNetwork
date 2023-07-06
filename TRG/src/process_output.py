import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time
# import jax.numpy as jnp
# from opt_einsum import contract
import concurrent.futures as cf

from TRG.src.dataclass import Input, Single_Result


def get_result(
    input: Input,
    pure: npt.NDArray[Any],
    impure_1: npt.NDArray[Any],
    impure_2: npt.NDArray[Any],
    impure_3: npt.NDArray[Any],
    impure_4: npt.NDArray[Any],
    pure_norms: list,
    first_order_norms: list[list],
    current_step: int,
    SIZE: int,
) -> Single_Result:

    single_result = get_free_energy(
        pure, pure_norms, current_step, SIZE)

    # order_parameter = get_order_parameter(
    #     pure, impure_1, impure_2, impure_3, impure_4,
    #     pure_norms, first_order_norms,
    #     current_step, SIZE
    # )

    # single_result.order_parameter = order_parameter

    return single_result


def get_free_energy(
    pure: npt.NDArray[Any],
    pure_norms: list,
    current_step: int,
    SIZE: int
) -> Single_Result:

    free_energy = np.log(np.abs(
        np.einsum("ijij->", pure, optimize=True))) / SIZE

    for j, pure_norm in zip(np.arange(current_step+1), pure_norms):
        remain = current_step-j
        free_energy += 2**remain*np.log(pure_norm) / SIZE

    return Single_Result(
        free_energy=free_energy,
        order_parameter=0.0,
        hamiltonian=0.0,
        heat_capacity=0.0,
    )


def get_order_parameter(
    pure: npt.NDArray[Any],
    impure_1: npt.NDArray[Any],
    impure_2: npt.NDArray[Any],
    impure_3: npt.NDArray[Any],
    impure_4: npt.NDArray[Any],
    pure_norms: list,
    first_order_norms: list,
    current_step: int,
    SIZE: int,
) -> float:

    order_parameter = efficient_contraction(impure_1, impure_2, impure_3, impure_4) / \
        efficient_contraction(pure, pure, pure, pure)

    for j, pure_norm in zip(
            np.arange(current_step+1),
            pure_norms
    ):
        order_parameter *= first_order_norms[0][j] * first_order_norms[1][j] * \
            first_order_norms[2][j] * first_order_norms[3][j] / pure_norm**4

    return np.real(order_parameter)


def efficient_contraction(
    T1: npt.NDArray[Any],
    T2: npt.NDArray[Any],
    T3: npt.NDArray[Any],
    T4: npt.NDArray[Any],
) -> Any:

    Up = np.einsum("abcd,cjkl->abjkld", T1, T4, optimize=True)
    Down = np.einsum("efgb,ghij->efhijb", T2, T3, optimize=True)

    return np.einsum("abjald,edlejb->", Up, Down, optimize=True)
