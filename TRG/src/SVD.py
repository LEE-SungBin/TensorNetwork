import numpy as np
import numpy.typing as npt
from typing import Any
import scipy as sp
import itertools
import time
# import jax.numpy as jnp
# from opt_einsum import contract

from TRG.src.dataclass import Mid_Time


def SVD(
    T: npt.NDArray[Any],
    left_indices: list[int],
    right_indices: list[int],
    Dcut: int,
    mid_time: Mid_Time,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:

    T = np.transpose(T, left_indices + right_indices)

    left_index_sizes = [T.shape[i] for i in range(len(left_indices))]
    right_index_sizes = [T.shape[i] for i in range(
        len(left_indices), len(left_indices) + len(right_indices))]
    xsize, ysize = int(np.prod(left_index_sizes)), int(
        np.prod(right_index_sizes))
    D = min(xsize, ysize, Dcut)

    now = time.perf_counter()
    T = T.reshape(xsize, ysize)
    mid_time.reshape.append(time.perf_counter()-now)

    now = time.perf_counter()
    U, S, Vh = np.linalg.svd(T, full_matrices=False)
    mid_time.decompose.append(time.perf_counter()-now)
    V, S = np.transpose(Vh), np.diag(np.sqrt(S[:D]))

    Dleft = []
    for size in left_index_sizes:
        Dleft.append(min(size, Dcut))
    Dright = []
    for size in right_index_sizes:
        Dright.append(min(size, Dcut))

    now = time.perf_counter()
    front_size = 1
    for i in range(len(left_index_sizes)+1):
        if i < len(left_index_sizes):
            U = U.reshape(front_size, left_index_sizes[i], -1)[:, :Dleft[i], :]
            front_size *= Dleft[i]
        else:
            U = U.reshape(front_size, -1)[:, :D]  # .reshape(-1, D)

    front_size = 1
    for i in range(len(right_index_sizes)+1):
        if i < len(right_index_sizes):
            V = V.reshape(
                front_size, right_index_sizes[i], -1)[:, :Dright[i], :]
            front_size *= Dright[i]
        else:
            V = V.reshape(front_size, -1)[:, :D]  # .reshape(-1, D)
    mid_time.reshape.append(time.perf_counter()-now)

    Dleft.append(D)
    Dright.insert(0, D)

    S1 = (U @ S).reshape(Dleft)
    S2 = np.transpose((V @ S)).reshape(Dright)

    return S1, S2  # * T approx S1 @ S2


def pure_TRG(
    pure: npt.NDArray[Any],
    Dcut: int,
    mid_time: Mid_Time
) -> tuple[npt.NDArray[Any], float]:

    pure_LU, pure_LV = SVD(pure, [3, 0], [1, 2], Dcut, mid_time)
    pure_RU, pure_RV = SVD(pure, [2, 3], [0, 1], Dcut, mid_time)

    now = time.perf_counter()
    new_pure = efficient_einsum(pure_LV, pure_RU, pure_LU, pure_RV)
    maximum = np.max(new_pure)
    new_pure /= maximum
    mid_time.truncate.append(time.perf_counter()-now)

    return new_pure, maximum


def impure_TRG(
    pure: npt.NDArray[Any],
    impure_1: npt.NDArray[Any],
    impure_2: npt.NDArray[Any],
    impure_3: npt.NDArray[Any],
    impure_4: npt.NDArray[Any],
    Dcut: int,
    mid_time: Mid_Time
) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        float, float, float, float
]:

    pure_LU, pure_LV = SVD(pure, [3, 0], [1, 2], Dcut, mid_time)
    pure_RU, pure_RV = SVD(pure, [2, 3], [0, 1], Dcut, mid_time)

    impure_1_RU, impure_1_RV = SVD(impure_1, [2, 3], [0, 1], Dcut, mid_time)
    impure_2_LU, impure_2_LV = SVD(impure_2, [3, 0], [1, 2], Dcut, mid_time)
    impure_3_RU, impure_3_RV = SVD(impure_3, [2, 3], [0, 1], Dcut, mid_time)
    impure_4_LU, impure_4_LV = SVD(impure_4, [3, 0], [1, 2], Dcut, mid_time)

    now = time.perf_counter()
    new_impure_1 = efficient_einsum(pure_LV, impure_1_RU, impure_4_LU, pure_RV)
    new_impure_2 = efficient_einsum(pure_LV, pure_RU, impure_2_LU, impure_1_RV)
    new_impure_3 = efficient_einsum(impure_2_LV, pure_RU, pure_LU, impure_3_RV)
    new_impure_4 = efficient_einsum(impure_4_LV, impure_3_RU, pure_LU, pure_RV)

    impure_max_1, impure_max_2, impure_max_3, impure_max_4 = (
        np.max(new_impure_1),
        np.max(new_impure_2),
        np.max(new_impure_3),
        np.max(new_impure_4),
    )

    new_impure_1, new_impure_2, new_impure_3, new_impure_4 = (
        new_impure_1 / impure_max_1,
        new_impure_2 / impure_max_2,
        new_impure_3 / impure_max_3,
        new_impure_4 / impure_max_4,
    )

    mid_time.truncate.append(time.perf_counter()-now)

    return (
        new_impure_1,
        new_impure_2,
        new_impure_3,
        new_impure_4,
        impure_max_1,
        impure_max_2,
        impure_max_3,
        impure_max_4
    )


def efficient_einsum(
    TLV: npt.NDArray[Any],
    TRU: npt.NDArray[Any],
    TLU: npt.NDArray[Any],
    TRV: npt.NDArray[Any],
) -> npt.NDArray[Any]:

    Up = np.einsum("iad,ldc->iacl", TLV, TRV, optimize=True)
    Down = np.einsum("baj,cbk->ajkc", TRU, TLU, optimize=True)

    return np.einsum("iacl,ajkc->ijkl", Up, Down, optimize=True)
