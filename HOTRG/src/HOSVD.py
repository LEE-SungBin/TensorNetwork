import numpy as np
import numpy.typing as npt
from typing import Any
import itertools
import time
# from tensorly.decomposition import tucker  # Higher-Order SVD
# from tensorly import tucker_to_tensor
from HOTRG.src.dataclass import Mid_Time


def HOSVD(
    lattice: npt.NDArray[np.float64],
    left_indices: list[int],
    right_indices: list[int],
    bond: int,
    mid_time: Mid_Time,
) -> tuple[npt.NDArray[np.float64], np.float64]:

    lattice = np.transpose(lattice, left_indices + right_indices)

    left_index_sizes = [lattice.shape[i] for i in range(len(left_indices))]
    right_index_sizes = [lattice.shape[i] for i in range(
        len(left_indices), len(left_indices) + len(right_indices))]
    xsize, ysize = np.prod(left_index_sizes), np.prod(right_index_sizes)

    now = time.perf_counter()
    lattice = lattice.reshape(xsize, ysize)
    mid_time.reshape.append(time.perf_counter()-now)

    now = time.perf_counter()
    M = np.matmul(lattice, lattice.T)
    mid_time.matmul.append(time.perf_counter()-now)

    now = time.perf_counter()
    # U, Lambda, _ = np.linalg.svd(M)

    # print(M)
    Lambda, U = np.linalg.eigh(M)
    U = np.flip(U, axis=(1,))
    Lambda = np.flip(Lambda, axis=(0,))
    # print(np.allclose(U@np.diag(Lambda)@U.T, M))
    mid_time.decompose.append(time.perf_counter()-now)

    return U, np.sum(Lambda[bond:])


def HOTRG_UD(
    pure: npt.NDArray[np.float64],
    first_order: npt.NDArray[np.complex128],
    Dcut: int,
    mid_time: Mid_Time,
) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.complex128],
        np.float64,
        np.complex128
]:

    originL, originD, originR, originU = pure.shape

    bondL, bondR = (
        min(originL**2, Dcut),
        min(originR**2, Dcut)
    )

    """
      d|
    a--T--c
      b|
      b|
    e--T--g
      f|
    """

    now = time.perf_counter()
    new_pure = np.einsum("abcd,efgb->aefgcd",
                         pure, pure, optimize=True)
    mid_time.create.append(time.perf_counter()-now)

    U_L, epsilon_L = HOSVD(new_pure, [0, 1], [2, 3, 4, 5], bondL, mid_time)
    U_R, epsilon_R = HOSVD(new_pure, [3, 4], [0, 1, 2, 5], bondR, mid_time)

    if epsilon_L <= epsilon_R:
        trunc_U = U_L.reshape(originL, originL, -1)[:, :, :bondL]
        trunc_U_T = U_L.T.reshape(-1, originL, originL)[:bondL, :, :]
    else:
        trunc_U = U_R.reshape(originR, originR, -1)[:, :, :bondR]
        trunc_U_T = U_R.T.reshape(-1, originR, originR)[:bondR, :, :]

    if np.allclose(np.diag([1.0 for i in range(bondL)]),
                   np.einsum("iae,aej->ij", trunc_U_T, trunc_U)) == False:
        raise ValueError("decomposition not correct")

    """
    U.T @ U = I even after truncation
    U @ U.T != I after truncation
    """

    now = time.perf_counter()

    trunc_pure = real_einsum(
        trunc_U_T, pure, trunc_U, pure, mode="UD")
    trunc_first_order = 1/2*(
        efficient_einsum(trunc_U_T, first_order, trunc_U, pure, mode="UD") +
        efficient_einsum(trunc_U_T, pure, trunc_U, first_order, mode="UD")
    )

    mid_time.truncate.append(time.perf_counter()-now)

    max_pure, max_first_order = np.max(trunc_pure), np.max(trunc_first_order)
    trunc_pure /= max_pure
    trunc_first_order /= max_first_order

    return trunc_pure, trunc_first_order, max_pure, max_first_order


def HOTRG_LR(
    pure: npt.NDArray[np.float64],
    first_order: npt.NDArray[np.complex128],
    Dcut: int,
    mid_time: Mid_Time,
) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.complex128],
        np.float64,
        np.complex128
]:

    originL, originD, originR, originU = pure.shape

    bondD, bondU = (
        min(originD**2, Dcut),
        min(originU**2, Dcut)
    )

    """
      d|      g|
    a--T--c c--T--f
      b|      e|
    """

    now = time.perf_counter()
    new_pure = np.einsum("abcd,cefg->abefgd",
                         pure, pure, optimize=True)
    mid_time.create.append(time.perf_counter()-now)

    U_D, epsilon_D = HOSVD(new_pure, [1, 2], [0, 3, 4, 5], bondD, mid_time)
    U_U, epsilon_U = HOSVD(new_pure, [4, 5], [0, 1, 2, 3], bondU, mid_time)

    if epsilon_D <= epsilon_U:
        trunc_U = U_D.reshape(originD, originD, -1)[:, :, :bondD]
        trunc_U_T = U_D.T.reshape(-1, originD, originD)[:bondD, :, :]
    else:
        trunc_U = U_U.reshape(originU, originU, -1)[:, :, :bondU]
        trunc_U_T = U_U.T.reshape(-1, originU, originU)[:bondU, :, :]

    if np.allclose(np.diag([1.0 for i in range(bondD)]),
                   np.einsum("iae,aej->ij", trunc_U_T, trunc_U)) == False:
        raise ValueError("decomposition not correct")

    """
    U.T @ U = I even after truncation
    U @ U.T != I after truncation
    """

    now = time.perf_counter()
    trunc_pure = real_einsum(
        trunc_U_T, pure, trunc_U, pure, mode="LR")
    trunc_first_order = 1/2*(
        efficient_einsum(trunc_U_T, first_order, trunc_U, pure, mode="LR") +
        efficient_einsum(trunc_U_T, pure, trunc_U, first_order, mode="LR")
    )

    mid_time.truncate.append(time.perf_counter()-now)

    max_pure, max_first_order = np.max(trunc_pure), np.max(trunc_first_order)
    trunc_pure /= max_pure
    trunc_first_order /= max_first_order

    return trunc_pure, trunc_first_order, max_pure, max_first_order


def efficient_einsum(
    T1: npt.NDArray[np.float64],
    T2: npt.NDArray[np.float64 | np.complex128],
    T3: npt.NDArray[np.float64],
    T4: npt.NDArray[np.float64 | np.complex128],
    mode: str,
) -> npt.NDArray[np.complex128]:

    if mode == "UD":
        left = np.einsum("iae,ejgb->ijgba", T1, T2, optimize=True)
        right = np.einsum("cgk,abcl->abgkl", T3, T4, optimize=True)

        return np.einsum("ijgba,abgkl->ijkl", left, right, optimize=True)

    elif mode == "LR":
        up = np.einsum("ldg,ibcd->libcg", T1, T2, optimize=True)
        down = np.einsum("bej,cekg->bcgkj", T3, T4, optimize=True)

        return np.einsum("libcg,bcgkj->ijkl", up, down, optimize=True)

    raise ValueError("mode must be 'UD' or 'LR'")


def real_einsum(
    T1: npt.NDArray[np.float64],
    T2: npt.NDArray[np.float64],
    T3: npt.NDArray[np.float64],
    T4: npt.NDArray[np.float64],
    mode: str,
) -> npt.NDArray[np.float64]:

    if mode == "UD":
        left = np.einsum("iae,ejgb->ijgba", T1, T2, optimize=True)
        right = np.einsum("cgk,abcl->abgkl", T3, T4, optimize=True)

        return np.einsum("ijgba,abgkl->ijkl", left, right, optimize=True)

    elif mode == "LR":
        up = np.einsum("ldg,ibcd->libcg", T1, T2, optimize=True)
        down = np.einsum("bej,cekg->bcgkj", T3, T4, optimize=True)

        return np.einsum("libcg,bcgkj->ijkl", up, down, optimize=True)

    raise ValueError("mode must be 'UD' or 'LR'")
