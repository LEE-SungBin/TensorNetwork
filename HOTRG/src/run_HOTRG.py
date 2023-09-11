import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time
# import jax.numpy as jnp
# from opt_einsum import contract
import sys

from lib.base_initial_state import get_pure_state, get_higher_order_moment
from HOTRG.src.HOSVD import HOTRG_UD, HOTRG_LR
from HOTRG.src.process_output import get_result
from HOTRG.src.dataclass import (
    Input, Setting, Result, Mid_Time, Time
)


def run_HOTRG(
    input: Input
) -> tuple[Setting, Result, Time]:

    # * RG step and Bond dimension
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
    mid_time = Mid_Time(initial=0.0, create=[], reshape=[], matmul=[],
                        decompose=[], truncate=[], total=[])

    SIZE = 1
    now = time.perf_counter()
    pure, first_order = get_pure_state(
        input), get_higher_order_moment(input, order=1)

    mid_time.initial = time.perf_counter()-now
    mid_time.total.append(time.perf_counter()-now)

    pure_norms_UD, pure_norms_LR = [], []
    first_order_norms_UD, first_order_norms_LR = [], []

    for current_step in range(step):
        SIZE *= 4
        now = time.perf_counter()
        """
          d|
        a--T--c
          b|
          b|
        e--T--g
          f|
        """

        # * HOTRG of up and down
        (temp_pure, temp_first_order,
         pure_max_UD, first_order_max_UD) = HOTRG_UD(
            pure, first_order, Dcut, mid_time)
        pure_norms_UD.append(pure_max_UD)
        first_order_norms_UD.append(first_order_max_UD)

        """
          d|      g|
        a--T--c c--T--f
          b|      e|
        """

        # * HOTRG of left and right
        (new_pure, new_first_order,
         pure_max_LR, first_order_max_LR) = HOTRG_LR(
            temp_pure, temp_first_order, Dcut, mid_time)
        pure_norms_LR.append(pure_max_LR)
        first_order_norms_LR.append(first_order_max_LR)

        """
        HOTRG 1 step finished, 2x2 -> 1x1
        """

        pure, first_order = new_pure, new_first_order

        single_result = get_result(
            input, pure, first_order,
            pure_norms_UD, pure_norms_LR,
            first_order_norms_UD, first_order_norms_LR,
            current_step, SIZE)

        result.free_energy.append(single_result.free_energy)
        result.hamiltonian.append(single_result.hamiltonian)
        result.heat_capacity.append(single_result.heat_capacity)
        result.order_parameter.append(
            single_result.order_parameter)

        mid_time.total.append(time.perf_counter()-now)

    return setting, result, mid_time.summarize_time()
