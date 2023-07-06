import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time
# import jax.numpy as jnp
# from opt_einsum import contract
# import scipy.integrate as integrate
import concurrent.futures as cf
from pathlib import Path

from TRG.src.run_TRG import run_TRG
from TRG.src.dataclass import Parameter, TRG, Input
from lib.base_manage_data import save_log, save_result


def multiprocessing(
    state: int,
    beta: npt.NDArray[np.float64],
    magnetic_field: npt.NDArray[np.float64],
    step: int,
    Dcut: int,
    max_workers: int,
) -> None:

    length = len(beta)
    inputs: list[Input] = []

    for i in range(length):
        parameter = Parameter(
            state=state, beta=beta[i], magnetic_field=magnetic_field[i])
        trg = TRG(step=step, Dcut=Dcut)
        inputs.append(Input(parameter=parameter, RG_operation=trg))

    finished = 0
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_TRG, input) for input in inputs
        ]

        for future in cf.as_completed(futures):
            finished += 1
            setting, result, tot_time = future.result()

            save_log(
                location=Path(__file__).parents[1],
                setting=setting,
                result=result,
                tot_time=tot_time
            )

            save_result(
                location=Path(__file__).parents[1],
                setting=setting,
                result=result,
                tot_time=tot_time)
            # print(finished, end=" ")
