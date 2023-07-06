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

# from abs_path import abs_path
# sys.path.append(f"{abs_path.abs_path}")


@dataclass
class Parameter(Base_Parameter):
    pass


@dataclass
class TRG(Base_RG_Operation):
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
    process: float

    @classmethod
    def from_base_time(cls, base_time: Base_Time, process: float) -> Time:
        kwargs = asdict(base_time)
        kwargs.update({"process": process})

        return cls(**kwargs)

    def to_log(self) -> str:
        return " ".join(
            f"{np.round(log, 3)}"
            for log in [
                self.initial,
                self.reshape,
                self.decompose,
                self.truncate,
                self.process,
                self.total,
            ]
        )


@dataclass
class Mid_Time(Base_Mid_Time):
    process: list

    def summarize_time(self) -> Time:
        base_time = super().summarize_time()
        process = np.sum(self.process)

        return Time.from_base_time(
            base_time=base_time,
            process=process
        )
