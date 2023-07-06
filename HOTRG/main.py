import argparse
import time
import matplotlib.pyplot as plt
from typing import Any
import numpy.typing as npt
import numpy as np
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[1]))
    print(f"{sys.path}")

    from HOTRG.src.multiprocessing import multiprocessing

    # sys.exit()

    parser = argparse.ArgumentParser()

    parser.add_argument("-Q", "--state", type=int, default=2)
    parser.add_argument("-N", "--Datapoint", type=int, default=100)
    parser.add_argument("-H", "--magnetic_field", type=float, default=0.0)
    parser.add_argument("-S", "--step", type=int, default=20)
    parser.add_argument("-D", "--Dcut", type=int, default=12)
    parser.add_argument("-max", "--max_workers", type=int, default=1)

    args = parser.parse_args()

    if (args.state == 0):  # XY model
        Kc = 1.12
    else:  # q-state potts model
        Kc = np.log(1+np.sqrt(args.state))*(args.state-1) / args.state

    coupling_max = np.round(2*Kc, 3)
    beta = np.linspace(start=coupling_max/args.Datapoint,
                       stop=coupling_max, num=args.Datapoint)[::-1]
    magnetic_field = np.full(args.Datapoint, args.magnetic_field)[::-1]

    multiprocessing(state=args.state, beta=beta, magnetic_field=magnetic_field,
                    step=args.step, Dcut=args.Dcut, max_workers=args.max_workers)
