import numpy as np
from tqdm import tqdm

import sim_cat_dtc_base


def collect_results(qnums=None, collect=False):
    if qnums is None:
        qnums = sim_cat_dtc_base.PAULI_ERROR_QNUM.keys()
    fids = np.zeros([len(qnums), 5])
    for qi, qnum in enumerate(qnums):
        for seed in tqdm(range(5), desc=f"{qnum}Q"):
            data_collect = sim_cat_dtc_base.run_ghz(
                seed, qnum, shots=30000, recal=False, collect=True
            )
            if collect:
                fids[qi, seed] = (
                    np.abs(data_collect["diag_list"].sum(axis=-1) / np.sqrt(2)) ** 2
                ).mean(axis=-1)
    if collect:
        return fids


if __name__ == "__main__":
    collect_results(np.arange(2,21,2))