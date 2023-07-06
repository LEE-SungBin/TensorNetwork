import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, asdict, field
from pathlib import PurePath
import pandas as pd
from typing import Any
import pickle


@dataclass
class Base_Parameter:
    state: int
    beta: float
    magnetic_field: float


@dataclass
class Base_RG_Operation:
    step: int
    Dcut: int


@dataclass
class Base_Single_Result:
    free_energy: float
    order_parameter: float
    hamiltonian: float
    heat_capacity: float


@dataclass
class Base_Result:
    free_energy: list
    order_parameter: list
    hamiltonian: list
    heat_capacity: list

    def to_log(self) -> str:
        return " ".join(
            f"{log}"
            for log in [
                np.round(self.free_energy[-1], 4),
                np.round(self.order_parameter[-1], 4),
                np.round(self.hamiltonian[-1], 4),
                np.round(self.heat_capacity[-1], 4),
            ]
        )


@dataclass
class Base_Input:
    parameter: Base_Parameter
    RG_operation: Base_RG_Operation


@dataclass
class Base_Setting:
    parameter: Base_Parameter
    RG_operation: Base_RG_Operation

    def to_log(self) -> str:
        return " ".join(
            f"{log}"
            for log in [
                self.parameter.state,
                np.round(self.parameter.beta, 3),
                np.round(self.parameter.magnetic_field, 3),
                self.RG_operation.step,
                self.RG_operation.Dcut,
            ]
        )


@dataclass
class Base_Time:
    initial: float
    reshape: float
    decompose: float
    truncate: float
    total: float

    def to_log(self) -> str:
        return " ".join(
            f"{np.round(log, 3)}"
            for log in [
                self.initial,
                self.reshape,
                self.decompose,
                self.truncate,
                self.total,
            ]
        )


@dataclass
class Base_Mid_Time:
    initial: float
    reshape: list
    decompose: list
    truncate: list
    total: list

    def summarize_time(self) -> Base_Time:
        initial = self.initial
        reshape = np.sum(self.reshape)
        decompose = np.sum(self.decompose)
        truncate = np.sum(self.truncate)
        total = np.sum(self.total)

        return Base_Time(
            initial=initial,
            reshape=reshape,
            decompose=decompose,
            truncate=truncate,
            total=total,
        )
