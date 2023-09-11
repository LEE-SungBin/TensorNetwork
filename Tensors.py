import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
from pathlib import Path
# * from typing import Self only available from python 3.11
# from typing import Self
from typing import Any
import hashlib
import pickle
import argparse
import time


@dataclass
class Quantum_State:
    """
    Clifford gates A: A @ sigma_1 @ A^T = sigma_2

    Gottesman-Knill theorem: Quantum circuits composed of stabilizer circuits, i.e., clifford gate + stabilizer input state + computational basis measurement
    can be simulated efficiently on a classical computer at most O(n^2m) cost for n-qubit circuit with m operations

    To achieve quantum supremacy, we have to use non-clifford gates as well as clifford gates
    """

    Identity = np.diag([1.0, 1.0])
    Hadamard_gate = np.array([
        [1.0, 1.0],
        [1.0, -1.0]
    ]) * 1.0/np.sqrt(2)
    sigma_x = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    sigma_y = np.array([
        [0.0, -1.0j],
        [1.0j, 0.0]
    ])
    sigma_z = np.diag([1.0, -1.0])
    S = np.array([1.0, 1.0j])
    CX_gate = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]).reshape(2, 2, 2, 2)
    CZ_gate = np.diag([1.0, 1.0, 1.0, -1.0]).reshape(2, 2, 2, 2)
    SWAP_gate = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]).reshape(2, 2, 2, 2)
    iSWAP_gate = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0j, 0.0],
        [0.0, 1.0j, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]).reshape(2, 2, 2, 2)

    """
    Non-Clifford gates: phase shift P for P(theta)|1>=exp(i theta)|1>, phase gate T, controlled phase S
    """

    T = np.diag([1.0, np.exp(np.pi/4*1j)])
    CS = np.diag([1.0, 1.0, 1.0, 1.0j]).reshape(2, 2, 2, 2)

    """
    Rotation operator gates: Rx(theta) = exp(-iX theta/2)
    """

    """
    Two qubit interaction gates, XX, YY, ZZ, and XX oplus YY interaction
    """

    def get_gate(self, gate: str) -> npt.NDArray[np.complex128]:
        if gate == "I" or gate == "i":
            return self.Identity
        elif gate == "H" or gate == "h":
            return self.Hadamard_gate
        elif gate == "X" or gate == "x":
            return self.sigma_x
        elif gate == "Y" or gate == "y":
            return self.sigma_y
        elif gate == "Z" or gate == "z":
            return self.sigma_z
        elif gate == "cx" or gate == "CX":
            return self.CX_gate
        elif gate == "swap" or gate == "SWAP":
            return self.SWAP_gate
        raise ValueError(f"{gate} not in the list")

    @staticmethod
    def MPS_to_tensor(
        MPS: list[npt.NDArray[np.complex128]]
    ) -> npt.NDArray[np.complex128]:

        contract = MPS[0]
        leg_dim = [MPS[0].shape[1]]

        for i, mps in enumerate(MPS):
            print(f"{i=} {mps.shape}")
            if i != 0:
                contract = np.tensordot(contract, mps, axes=[(-1), (0)])
                leg_dim.append(mps.shape[1])

        return contract.reshape(leg_dim)

    @staticmethod
    def tensor_to_MPS(
        tensor: npt.NDArray[np.complex128],
        max_bond: int
    ) -> list[npt.NDArray[np.complex128]]:

        MPS: list[npt.NDArray[np.complex128]] = []

        N_qubits = len(tensor.shape)
        rest = tensor
        bond_dim = 1

        for i, leg_dim in enumerate(tensor.shape):
            # print(f"{i=} {bond_dim=} {leg_dim=} {rest.shape=}")
            rest = rest.reshape(bond_dim*leg_dim, -1)

            U, S, Vh = np.linalg.svd(
                rest, full_matrices=False, hermitian=False)  # * rest = U @ S @ Vh
            assert np.allclose(
                rest, U @ np.diag(S) @ Vh), f"SVD error {np.max(np.abs(rest - U @ np.diag(S) @ Vh))}"
            U, S, Vh = U[:, :max_bond], np.diag(S[:max_bond]), Vh[:max_bond, :]

            MPS.append(U.reshape(bond_dim, leg_dim, S.shape[0]))
            rest = S @ Vh
            bond_dim = rest.shape[0]

            if i == N_qubits - 2:
                MPS.append(rest.reshape(rest.shape[0], rest.shape[1], 1))
                break

        return MPS

    @staticmethod
    def get_fidelity(tensor: npt.NDArray, MPS: list) -> float:

        tensor_from_mps = Quantum_State.MPS_to_tensor(MPS)

        assert len(tensor.shape) == len(MPS), f"{tensor.shape} != {len(MPS)}"

        N_qubits = len(tensor.shape)
        qubit_list = tuple(np.arange(N_qubits))

        raw = np.tensordot(tensor, np.conjugate(
            tensor_from_mps), axes=[qubit_list, qubit_list])

        return np.real(raw * np.conjugate(raw))


@dataclass
class Qubit:
    def __init__(self, thetas: list[float], phis: list[float]) -> None:

        assert len(thetas) == len(
            phis), f"len(thetas) {len(thetas)} != len(phis) {len(phis)}"
        self.N_qubits = len(thetas)

        self.get_qubits(thetas, phis)

    def get_qubits(self, thetas: list[float], phis: list[float]) -> None:

        self.qubits: list[npt.NDArray[np.complex128]] = []

        for theta, phi in zip(thetas, phis):
            assert (0.0 <= theta) & (
                theta <= np.pi), f"theta = {theta} != [0, pi]"
            assert (0.0 <= phi) & (
                phi <= 2 * np.pi), f"phi = {phi} != [0, 2*pi]"
            self.qubits.append(np.array([
                np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)
            ]))


class Tensor(Quantum_State):
    def __init__(self, qubit: Qubit) -> None:
        super().__init__()

        self.N_qubits = qubit.N_qubits
        self.qubits = qubit.qubits
        self.get_tensors()

    def get_tensors(self) -> None:
        self.tensor: npt.NDArray[np.complex128] = self.qubits[0]

        for i, qubit in enumerate(self.qubits):
            if i > 0:
                self.tensor = np.tensordot(
                    self.tensor, qubit, axes=0)  # * Tensor Product

        assert self.N_qubits == len(
            self.tensor.shape), f"tensor shape {self.tensor.shape} length != {self.N_qubits}"

        print(f"qubits = {self.qubits}")
        self.print_shape()
        self.print_all()

    def print_all(self) -> None:
        print(f"Tensor = {self.tensor.reshape(-1)}")

    def print_shape(self) -> None:
        print(f"Tensor shape = {self.tensor.shape}")

    def apply_single_gate(self, gates: list[str]) -> None:

        assert len(
            gates) == self.N_qubits, f"gates length {len(gates)} != {self.N_qubits}"

        tensor_operation = self.get_gate(gates[0])

        for i, gate in enumerate(gates):
            if i > 0:
                tensor_operation = np.tensordot(
                    tensor_operation, self.get_gate(gate), axes=0)

        self.tensor = np.tensordot(
            tensor_operation, self.tensor, axes=[tuple(2*np.arange(self.N_qubits)+1), tuple(np.arange(self.N_qubits))])

    def apply_two_qubit_gate(
        self,
        gates: list[str],
        controls: list[int],
        acts: list[int]
    ) -> None:

        assert len(gates) == len(controls) == len(
            acts), f"controls {len(controls)} != acts {len(acts)}"

        for gate, control, act in zip(gates, controls, acts):
            assert control != act, f"control {control} == act {act}"

            # * indices requires transpose
            indices = [i for i in range(self.N_qubits - 2)]

            if control < act:
                indices.insert(control, self.N_qubits - 2)
                indices.insert(act, self.N_qubits - 1)

            elif act < control:
                indices.insert(act, self.N_qubits - 1)
                indices.insert(control, self.N_qubits - 2)

            # * np.tensordot(A,B,axes=[(j), (j)]), A[i,j,k,l], B[r,j,k,t] -> [i,k,r,t]
            self.tensor = np.tensordot(
                self.tensor, self.get_gate(gate), axes=[(control, act), (2, 3)]
            ).transpose(indices)


class MPS(Quantum_State):

    def __init__(self, qubit: Qubit, max_bond: int):
        super().__init__()

        self.max_bond = max_bond
        self.N_qubits = qubit.N_qubits
        self.qubits = qubit.qubits
        self.get_MPS()

    def get_MPS(self) -> None:
        self.MPS: list[npt.NDArray] = []

        for qubit in self.qubits:
            self.MPS.append(qubit.reshape(1, -1, 1))

        print(f"qubits = {self.qubits}")
        self.print_shape()
        self.print_all()

    def print_all(self) -> None:
        print(f"MPS = {self.MPS}")

    def print_shape(self) -> None:
        for i, mps in enumerate(self.MPS):
            print(f"MPS {i} shape = {mps.shape}")

    def apply_single_gate(self, gates: list[str]) -> None:

        assert len(
            gates) == self.N_qubits, f"gates length {len(gates)} != {self.N_qubits}"

        for i, gate in enumerate(gates):
            # * tensordot requires indices transposition
            self.MPS[i] = np.tensordot(
                self.MPS[i], self.get_gate(gate), axes=[(1), (1)]).transpose([0, 2, 1])

        self.print_shape()

    def apply_two_qubit_gate(
            self,
            gates: list[str],
            controls: list[int],
            acts: list[int]
    ) -> None:

        assert len(gates) == len(controls) == len(
            acts), f"gates {len(gates)} controls {len(controls)} acts {len(acts)} do not match"

        for gate, control, act in zip(gates, controls, acts):
            assert np.abs(control-act) == 1, f"cx only for nn qubits"
            temp: npt.NDArray = np.einsum(
                "abcd,icj,jdk->iabk", self.get_gate(gate), self.MPS[control], self.MPS[act])

            assert temp.shape[1] == temp.shape[
                2] == 2, f"qubit dim {temp.shape[1]} {temp.shape[2]} != 2"

            bond_L, bond_R = temp.shape[0], temp.shape[3]
            temp = temp.reshape(bond_L*2, 2*bond_R)

            U, S, Vh = np.linalg.svd(temp.reshape(
                bond_L*2, 2*bond_R), full_matrices=False)
            assert np.allclose(
                temp, U @ np.diag(S) @ Vh), f"SVD error {np.max(np.abs(temp - U @ np.diag(S) @ Vh))}"
            U, S, Vh = U[:, :self.max_bond], np.diag(
                S[:self.max_bond]), Vh[:self.max_bond, :]

            self.MPS[control] = (U @ np.sqrt(S)).reshape(bond_L, 2, -1)
            self.MPS[act] = (np.sqrt(S) @ Vh).reshape(-1, 2, bond_R)

        self.print_shape()
