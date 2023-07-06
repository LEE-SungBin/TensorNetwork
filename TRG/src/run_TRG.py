import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time
import jax.numpy as jnp
from opt_einsum import contract
import concurrent.futures as cf

from lib.base_initial_state import get_pure_state, get_higher_order_moment
from TRG.src.SVD import pure_TRG, impure_TRG
from TRG.src.process_output import get_result
from TRG.src.dataclass import Input, Setting, Result, Mid_Time, Time


def run_TRG(
    input: Input
) -> tuple[Setting, Result, Time]:

    step, Dcut = (
        input.RG_operation.step,
        input.RG_operation.Dcut,
    )

    setting = Setting(
        parameter=input.parameter,
        RG_operation=input.RG_operation,
    )

    result = Result(free_energy=[], order_parameter=[],
                    hamiltonian=[], heat_capacity=[])
    mid_time = Mid_Time(initial=0.0, reshape=[], decompose=[],
                        truncate=[], process=[], total=[])

    SIZE = 1

    now = time.perf_counter()
    pure, first_order = get_pure_state(
        input), get_higher_order_moment(input, order=1)
    mid_time.initial = time.perf_counter()-now
    mid_time.total.append(time.perf_counter()-now)

    impure_1, impure_2, impure_3, impure_4 = first_order, pure, pure, pure

    pure_norms: list = []
    first_order_norms: list[list] = [[] for _ in range(4)]

    for current_step in range(step):
        SIZE *= 2
        now = time.perf_counter()
        """
          d|     l|
        a--T--cc--T--k
          b|     j|
          b|     j|
        e--T--gg--T--i
          f|     h|
        """
        new_pure, pure_max = pure_TRG(pure, Dcut, mid_time)
        pure_norms.append(pure_max)

        # (new_impure_1, new_impure_2, new_impure_3, new_impure_4,
        #  impure_max_1, impure_max_2, impure_max_3, impure_max_4) = impure_TRG(
        #     pure, impure_1, impure_2, impure_3, impure_4, Dcut, mid_time)

        # first_order_norms[0].append(impure_max_1)
        # first_order_norms[1].append(impure_max_2)
        # first_order_norms[2].append(impure_max_3)
        # first_order_norms[3].append(impure_max_4)

        """
        TRG 1 step finished, 2 -> 1
        """

        pure = new_pure

        # impure_1, impure_2, impure_3, impure_4 = (
        #     new_impure_1, new_impure_2, new_impure_3, new_impure_4
        # )

        start = time.perf_counter()

        single_result = get_result(
            input, pure, impure_1, impure_2, impure_3, impure_4,
            pure_norms, first_order_norms,
            current_step, SIZE
        )

        mid_time.process.append(time.perf_counter()-start)

        result.free_energy.append(single_result.free_energy)
        result.hamiltonian.append(single_result.hamiltonian)
        result.heat_capacity.append(single_result.heat_capacity)
        result.order_parameter.append(single_result.order_parameter)

        mid_time.total.append(time.perf_counter()-now)

    return setting, result, mid_time.summarize_time()
