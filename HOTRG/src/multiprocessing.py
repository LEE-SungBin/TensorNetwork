import numpy as np
import numpy.typing as npt
from typing import Any
from pathlib import Path
import scipy as sp
import time
# import jax.numpy as jnp
import concurrent.futures as cf

from lib.base_manage_data import save_log, save_result

from HOTRG.src.run_HOTRG import run_HOTRG
from HOTRG.src.dataclass import Parameter, HOTRG, Input


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
        hotrg = HOTRG(step=step, Dcut=Dcut)
        inputs.append(Input(parameter=parameter, RG_operation=hotrg))

    finished = 0
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_HOTRG, input) for input in inputs
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
                tot_time=tot_time
            )
            # print(finished, end=" ")
