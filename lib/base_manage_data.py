from __future__ import annotations

import hashlib
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import numpy.typing as npt
import json
import pickle
import pandas as pd
# from sklearn.linear_model import LinearRegression

from lib.base_dataclass import Base_Setting, Base_Result, Base_Time


def save_result(
    setting: Base_Setting,
    result: Base_Result,
    tot_time: Base_Time,
    location: Path = Path("."),
) -> None:
    key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    # key = "".join(
    #     np.random.choice(list(string.ascii_lowercase + string.digits), 6)
    # )

    data_path = location / "data"
    setting_path = location / "setting"

    data_path.mkdir(parents=True, exist_ok=True)
    setting_path.mkdir(parents=True, exist_ok=True)

    output = {
        "key": key,
    }

    output.update(asdict(setting.parameter))
    output.update(asdict(setting.RG_operation))

    # name = f"{setting.parameter.state}-state {setting.hotrg.Dcut} bond {key}"
    name = key

    with open(setting_path / f"{name}.json", "w") as file:
        json.dump(output, file)

    output.update(asdict(tot_time))
    output.update(asdict(result))

    with open(data_path / f"{name}.pkl", "wb") as file:
        pickle.dump(output, file)


def save_log(
    setting: Base_Setting,
    result: Base_Result,
    tot_time: Base_Time,
    location: Path = Path("."),
) -> None:

    log = (
        f"{datetime.now().replace(microsecond=0)} {setting.to_log()} {tot_time.to_log()} {result.to_log()}\n"
    )

    log_path = location
    # log_path.mkdir(parents=True, exist_ok=True)

    with open(log_path / "log.txt", "a") as file:
        file.write(log)


def get_setting(
    location: Path = Path("."),
    state: int | None = None,
    step: int | None = None,
    Dcut: int | None = None,
) -> Any:

    conditions: list[str] = []
    if state is not None:
        conditions.append(f"state == {state}")
    if step is not None:
        conditions.append(f"step == {step}")
    if Dcut is not None:
        conditions.append(f"Dcut == {Dcut}")

    def filter_file(f: Path) -> bool:
        return f.is_file() and (f.suffix == ".json") and f.stat().st_size > 0

    # * Scan the setting directory and gather result files
    setting_dir = location / f"./setting"
    setting_files = [f for f in setting_dir.iterdir() if filter_file(f)]

    # * Read files
    settings: list[dict[str, Any]] = []
    for file in setting_files:
        with open(file, "rb") as f:
            settings.append(json.load(f))

    df = pd.DataFrame(settings)

    return df.query(" and ".join(conditions))["key"]


def load_result(
    location: Path = Path("."),
    state: int | None = None,
    step: int | None = None,
    Dcut: int | None = None,
) -> pd.DataFrame:

    # * Scan the result directory and gather result files
    result_dir = location / f"data"
    result_keys = get_setting(location, state, step, Dcut)
    result_files = [result_dir /
                    f"{result_key}.pkl" for result_key in result_keys]

    # * Read files
    results: list[dict[str, Any]] = []
    for file in result_files:
        with open(file, "rb") as f:
            results.append(pickle.load(f))

    # * Concatenate to single dataframe
    df = pd.DataFrame(results)
    return df


def delete_result(
    key_names: list[str],
    location: Path = Path("."),
) -> None:
    del_setting, del_data = 0, 0

    for key in key_names:
        target_setting = location / f"setting/{key}.json"
        target_file = location / f"/data/{key}.pkl"

        try:
            target_setting.unlink()
            del_setting += 1
        except OSError:
            print(f"No setting found for key in setting: {key}")

        try:
            target_file.unlink()
            del_data += 1
        except OSError:
            print(f"No file found for key in data: {key}")

    print(f"setting deleted: {del_setting}, data deleted: {del_data}")


def delete_all(location: Path = Path("."),) -> None:
    setting_dir = location / "setting"
    data_dir = location / "data"

    settings = [f for f in setting_dir.iterdir()]
    datas = [f for f in data_dir.iterdir()]

    del_setting, del_data = 0, 0
    for setting in settings:
        try:
            setting.unlink()
            del_setting += 1
        except OSError:
            print(f"No setting found for key in setting: {setting}")
    for data in datas:
        try:
            data.unlink()
            del_data += 1
        except OSError:
            print(f"No file found for key in data: {data}")

    print(f"setting deleted: {del_setting}, data deleted: {del_data}")
