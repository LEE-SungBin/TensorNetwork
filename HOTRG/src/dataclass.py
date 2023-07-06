from __future__ import annotations

from lib.base_dataclass import (
    Base_Parameter, Base_RG_Operation,
    Base_Single_Result, Base_Result,
    Base_Input, Base_Setting,
    Base_Mid_Time, Base_Time
)

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, asdict, field
from pathlib import PurePath
import pandas as pd
from typing import Any
import pickle
import sys


@dataclass
class Parameter(Base_Parameter):
    pass


@dataclass
class HOTRG(Base_RG_Operation):
    pass


@dataclass
class Result(Base_Result):
    pass


@dataclass
class Single_Result(Base_Single_Result):
    pass


@dataclass
class Input(Base_Input):
    pass


@dataclass
class Setting(Base_Setting):
    pass


@dataclass
class Time(Base_Time):
    create: float
    matmul: float

    @classmethod
    def from_base_time(cls, base_time: Base_Time, create: float, matmul: float) -> Time:
        kwargs = asdict(base_time)
        kwargs.update({"create": create, "matmul": matmul})

        return cls(**kwargs)

    def to_log(self) -> str:
        return " ".join(
            f"{np.round(log, 3)}"  # for log in asdict(self).values()
            for log in [
                self.initial,
                self.create,
                self.reshape,
                self.matmul,
                self.decompose,
                self.truncate,
                self.total,
            ]
        )


@dataclass
class Mid_Time(Base_Mid_Time):
    create: list
    matmul: list

    def summarize_time(self) -> Time:
        base_time = super().summarize_time()
        create = np.sum(self.create)
        matmul = np.sum(self.matmul)

        return Time.from_base_time(
            base_time=base_time,
            create=create,
            matmul=matmul
        )

        # reshape = np.sum(self.reshape)
        # decompose = np.sum(self.decompose)
        # truncate = np.sum(self.truncate)
        # total = np.sum(self.total)

        return Time(
            create=create,
            reshape=reshape,
            matmul=matmul,
            decompose=decompose,
            truncate=truncate,
            total=total,
        )
