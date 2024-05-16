from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import time
import warnings

import numpy as np
import scipy

from deepdiff import DeepHash
import gen_circuit as gc

c_thread = 1
os.environ["OMP_NUM_THREADS"] = "1"

WARNING_MESSAGE = {}
WARNING_MESSAGE["old_mindquantum_deco"] = False

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mindquantum.core as mq_core
    import mindquantum.simulator as mq_simu

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CPU_DEFAULT = multiprocessing.cpu_count()
__ideal_GHZ_statevector__ = {}

PAULI_ERROR_QNUM = {
    2: {"sq": 0.049e-2, "cz": 0.256e-2},
    4: {"sq": 0.031e-2, "cz": 0.148e-2},
    6: {"sq": 0.045e-2, "cz": 0.199e-2},
    8: {"sq": 0.045e-2, "cz": 0.199e-2},
    10: {"sq": 0.064e-2, "cz": 0.225e-2},
    12: {"sq": 0.064e-2, "cz": 0.225e-2},
    14: {"sq": 0.064e-2, "cz": 0.225e-2},
    16: {"sq": 0.056e-2, "cz": 0.227e-2},
    18: {"sq": 0.056e-2, "cz": 0.227e-2},
    20: {"sq": 0.056e-2, "cz": 0.227e-2},
}

DEFAULT_NOISE_PARAMS = {
    "t1": 131.3e3,
    "t2_se": 33.8e3,
    "sq_len": {"i_sq": 24, "i_cz": 60, "tbuffer": 120, "PVZ": 24},
    "tq_len": {"cz": 60},
    "basis_gates": ["i_sq", "i_cz", "cz", "tbuffer", "PVZ"],
}


def run_ghz(
    seed,
    qnum,
    pool_size=CPU_DEFAULT // 2,
    shots=30000,
    recal=False,
    collect=False,
):
    noise_param = deepcopy(DEFAULT_NOISE_PARAMS)
    noise_param["sq_pauli_error"] = {"PVZ": PAULI_ERROR_QNUM[qnum]["sq"]}
    noise_param["tq_pauli_error"] = {"cz": PAULI_ERROR_QNUM[qnum]["cz"]}
    catdtc = CatDtcSim(qnum=qnum)
    catdtc.gen_noise_model_mindquantum(noise_param=noise_param)
    rst = catdtc.sim_diag_mindquantum(
        shots=shots,
        seed=seed,
        pool_size=pool_size,
        recal=recal,
    )
    if collect:
        return rst


class CatDtcSim:
    def __init__(
        self,
        qnum=4,
        q_names=None,
    ):
        self.q_names = gc.q_names[qnum] if q_names is None else q_names
        self.cat_pattern = gc.cat_pattern_sensing[qnum]
        self.c_names = [
            n for ns in list(self.cat_pattern.values()) for n in ns if n.startswith("c")
        ]
        self.qnum = len(self.q_names)
        self.qnum_str = str(qnum)
        self.isPlot = True
        self.circuit = None
        self.simulator = "qiskit"

    def reset_noise(self):
        self.noise_model = None
        self.hash_val = None
        self.circuit = None
        self.simulator = None

    def gen_noise_model_mindquantum(self, noise_param, noisy_message_level=1):
        self.noise_model, self.hash_val = gen_noise_channels(
            self.q_names,
            self.c_names,
            noise_param=noise_param,
            noisy_message_level=noisy_message_level,
        )
        self.simulator = "mindquantum"
        return self.noise_model, self.hash_val

    def gen_file_name(
        self,
        shots,
        seed,
        hash_val,
        exp_name="GHZ",
        sub_folder="",
        save_name="",
        **kwargs,
    ):
        file_name = f"{FILE_DIR}/{self.simulator}/"
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        file_name += f"{exp_name}/"
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        if hash_val is None:
            file_name += f"Q{self.qnum_str}_ideal/"
        else:
            file_name += f"Q{self.qnum_str}_noise_model={hash_val[:8]}/"
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        if sub_folder:
            file_name += f"{sub_folder}/"
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        file_name += f"shots={shots}_seed={seed}_{save_name}"
        for key, val in kwargs.items():
            file_name += f"_{key}={val}"
        return file_name + ".mat"

    def sim_diag_mindquantum(
        self,
        shots,
        seed,
        recal=False,
        noisy=False,
        pool_size=5,
        save_name="",
    ):
        file_name = self.gen_file_name(
            shots, seed, self.hash_val, exp_name="diag", save_name=save_name
        )
        self.file_name = file_name
        if recal is None:
            assert Path(file_name).exists(), file_name
        cir_dict = gc.gen_ghz_circuit(gc.cat_pattern_sensing[self.qnum])
        q_cir = gen_mindquantum_circuit(cir_dict, self.q_names, is_pvz=True)
        self.circuit = q_cir
        if Path(file_name).exists() and not recal:
            rst = scipy.io.loadmat(file_name, simplify_cells=True)
            if noisy:
                print(f"load {self.qnum}-Q diag")
            return rst
        t0 = time.time()
        rst = {}
        if self.hash_val:
            seq_adder = mq_core.SequentialAdder(self.noise_model)
            cir_sim = seq_adder(q_cir)
        else:
            cir_sim = q_cir
        idxs = np.array(
            [int(idx_str * (self.qnum // 2), base=2) for idx_str in ["01", "10"]]
        )
        diag_list = run_mqvector_shots_idx(
            cir_sim, seed, shots, pool_size=pool_size, idxs=idxs
        )
        rst.update({"diag_list": diag_list})
        rst["sim_time"] = time.time() - t0
        Path(file_name).parent.mkdir() if not Path(file_name).parent.exists() else None
        scipy.io.savemat(file_name, rst)
        return rst


class PauliErrorAdder(mq_core.ChannelAdderBase):
    def __init__(self, qidx, pauli_error, gate_names=["RX", "RY"], noisy=False):
        self.q = qidx if isinstance(qidx, (list, tuple)) else [qidx]
        self.num_qs = len(self.q)
        self.pauli_error = pauli_error
        self.p = pauli_error / (1 - 1 / 4**self.num_qs)
        self.gate_names = gate_names
        self.noisy = noisy
        super().__init__()

    def _accepter(self):
        return [
            lambda x: set(self.q) == set(x.obj_qubits + x.ctrl_qubits)
            and (x.name in self.gate_names)
        ]

    def _excluder(self):
        return [
            lambda x: isinstance(
                x, (mq_core.gates.Measure, mq_core.NoiseGate, mq_core.IGate)
            )
        ]

    def _handler(self, g):
        if self.noisy:
            print(f"add control noise after", g, f": {self}")
        return mq_core.Circuit(
            [mq_core.DepolarizingChannel(self.p, self.num_qs).on(self.q)]
        )

    def __repr__(self):
        return f"PauliErrorAdder<q={self.q}, pauli_error={self.pauli_error}>"


class DecoherenceAdder(mq_core.ChannelAdderBase):
    def __init__(
        self,
        qidx,
        T1,
        T2,
        gate_length=50,
        gate_names=["RX", "RY", "RZ", "Z"],
        noisy=False,
    ):
        self.q = qidx if isinstance(qidx, (list, tuple)) else [qidx]
        self.num_qs = len(self.q)
        self.T1 = T1
        self.T2 = T2
        self.gate_length = gate_length
        self.p1 = 1 - np.exp(-gate_length / T1)
        self.p2 = 1 - np.exp(-gate_length / (T2 / 2) + gate_length / T1)
        self.gate_names = gate_names
        self.noisy = noisy
        super().__init__()

    def _accepter(self):
        return [
            lambda x: set(self.q) == set(x.obj_qubits + x.ctrl_qubits)
            and (x.name in self.gate_names)
        ]

    def _excluder(self):
        return [lambda x: isinstance(x, (mq_core.gates.Measure, mq_core.NoiseGate))]

    def _handler(self, g):
        if self.noisy:
            print(f"add thermal noise after", g, f": {self}")
        try:
            # mq_core.
            channel = np.hstack(
                [
                    [
                        mq_core.ThermalRelaxationChannel(
                            self.T1, self.T2, self.gate_length
                        ).on(q_i)
                    ]
                    for q_i in self.q
                ]
            ).tolist()
        except:
            if not WARNING_MESSAGE["old_mindquantum_deco"]:
                print("using old version of ADC & PDC")
                WARNING_MESSAGE["old_mindquantum_deco"] = True
            channel = np.hstack(
                [
                    [
                        mq_core.AmplitudeDampingChannel(self.p1).on(q_i),
                        mq_core.PhaseDampingChannel(self.p2).on(q_i),
                    ]
                    for q_i in self.q
                ]
            ).tolist()
        return mq_core.Circuit(channel)

    def __repr__(self):
        return f"ThermalRelaxationChannel<q={self.q}, T1={self.T1}, T2*={self.T2}>"


def pvz(alpha, theta, phi):
    return np.array(
        [
            [
                np.cos(alpha / 2),
                -1j
                * np.sin(alpha / 2)
                * (np.cos(theta) - 1j * np.sin(theta))
                * np.exp(1j * phi),
            ],
            [
                -1j * np.sin(alpha / 2) * (np.cos(theta) + 1j * np.sin(theta)),
                np.cos(alpha / 2) * np.exp(1j * phi),
            ],
        ]
    )


def i_gate():
    return np.array([[1, 0], [0, 1]])


def gen_mindquantum_circuit(dic_circuit, q_names, dtc_params={}, is_pvz=True):
    """
    gates: ['i_reset', 'i_cz', 'i_sq', 'pvz', 'cz', 'tbuffer']
    """

    # load dtc params
    lmbda = dtc_params.get("lmbda", 0.05)
    theta = dtc_params.get("theta", 0.05)
    lmbda_phi = dtc_params.get("lmbda_phi", 0)
    hz = dtc_params.get("hz", 0.3)
    xi = dtc_params.get("xi", 0)
    gamma = dtc_params.get("gamma", 0)
    # begin circuit
    num_qs = len(q_names)
    circuit = mq_core.Circuit()
    for i in range(len(q_names)):
        circuit.append(mq_core.gates.UnivMathGate("i_reset", i_gate())(i))
    circuit.barrier()
    layers_circuit = len(list(dic_circuit.keys()))
    for layer_pre in np.arange(0, layers_circuit, 1):
        layer_qs = []
        for q_name, gate in dic_circuit[layer_pre].items():
            if q_name.startswith("q"):
                layer_qs.append(q_name)
                if gate == []:
                    circuit.append(
                        mq_core.gates.UnivMathGate("i_sq", i_gate())(
                            q_names.index(q_name)
                        )
                    )
                if is_pvz:
                    sq_rc = gc.RotationCompiler()
                    for q_op in gate:
                        if isinstance(q_op, str):
                            sq_rc.plus_gate(q_op)
                        elif isinstance(q_op, tuple):
                            if isinstance(q_op[1], str):
                                phase_tmp = eval(q_op[1])
                            else:
                                phase_tmp = q_op[1]
                            sq_rc.__getattribute__(q_op[0])(phase_tmp)
                        else:
                            raise ValueError(
                                f"Unknow gate {q_op} for qubit {q_name} in layer {layer_pre}"
                            )
                    unitary_mat = sq_rc.compile_gate()
                    circuit.append(
                        mq_core.gates.UnivMathGate("PVZ", unitary_mat)(
                            q_names.index(q_name)
                        )
                    )
                else:
                    for q_op in gate:
                        if isinstance(q_op, str):
                            circuit.__getattribute__(q_op.lower())(
                                q_names.index(q_name)
                            )
                        elif isinstance(q_op, tuple):
                            if isinstance(q_op[1], str):
                                phase_tmp = eval(q_op[1])
                            else:
                                phase_tmp = q_op[1]
                            circuit.__getattribute__(q_op[0].lower())(
                                phase_tmp, q_names.index(q_name)
                            )
                        else:
                            raise ValueError(
                                f"Unknow gate {q_op} for qubit {q_name} in layer {layer_pre}"
                            )
                layer_type = "i_sq"
            elif q_name.startswith("c"):
                cz_qs = gc.c2q(q_name)
                layer_qs.extend(cz_qs)
                for q_op in gate:
                    if q_op == "cz":
                        circuit.append(
                            mq_core.Z(q_names.index(cz_qs[0]), q_names.index(cz_qs[1]))
                        )
                layer_type = "i_cz"
            elif q_name == "tbuffer":
                layer_type = "tbuffer"

        for q_name in q_names:
            if q_name not in layer_qs:
                circuit.append(
                    mq_core.gates.UnivMathGate(layer_type, i_gate())(
                        q_names.index(q_name)
                    )
                )
        circuit.barrier()
    return circuit


def add_thermal_noise_mindquantum(
    noise_channels,
    q_names,
    c_names,
    t1,
    t2_se,
    sq_gate_len_dict=None,
    tq_gate_len_dict=None,
    noisy_message_level=1,
):
    if noisy_message_level > 1:
        print("add thermal noise: ", sq_gate_len_dict, tq_gate_len_dict)
    noise_channels_add = []
    for gate_op, gate_len in sq_gate_len_dict.items():
        for q_i, q in enumerate(q_names):
            T1 = t1[q] if isinstance(t1, dict) else t1
            T2_SE = t2_se[q] if isinstance(t2_se, dict) else t2_se
            T2 = 1 / (1 / (2 * T1) + 1 / T2_SE)
            noise_channels_add.append(
                DecoherenceAdder(
                    qidx=q_i, T1=T1, T2=T2, gate_length=gate_len, gate_names=[gate_op]
                ),
            )

    for gate_op, gate_len in tq_gate_len_dict.items():
        for c_i, c in enumerate(c_names):
            qs = gc.c2q(c)
            q_idxes = [q_names.index(q) for q in qs]
            T1 = t1[q] if isinstance(t1, dict) else t1
            T2_SE = t2_se[q] if isinstance(t2_se, dict) else t2_se
            T2 = 1 / (1 / (2 * T1) + 1 / T2_SE)
            noise_channels_add.append(
                DecoherenceAdder(
                    qidx=q_idxes,
                    T1=T1,
                    T2=T2,
                    gate_length=gate_len,
                    gate_names=[gate_op],
                ),
            )
    noise_channels += noise_channels_add


def add_sq_pauli_error_mindquantum(
    noise_channels, q_names, error_dict={}, noisy_message_level=1
):
    if noisy_message_level > 1:
        print("add single-qubit control error: ", error_dict)
    noise_channels_add = []
    for gate_op, ep in error_dict.items():
        for q_i, q in enumerate(q_names):
            noise_channels_add.append(
                PauliErrorAdder(qidx=q_i, pauli_error=ep, gate_names=[gate_op])
            )
            if noisy_message_level > 2:
                print(f"Add {q}.{gate_op} pauli error {ep}")
    noise_channels += noise_channels_add


def add_tq_pauli_error_mindquantum(
    noise_channels, q_names, c_names, error_dict={}, noisy_message_level=1
):
    if noisy_message_level > 1:
        print("add two-qubit control error: ", error_dict)
    noise_channels_add = []
    for gate_op, ep in error_dict.items():
        for c_i, c in enumerate(c_names):
            qs = gc.c2q(c)
            q_idxes = [q_names.index(q) for q in qs]
            noise_channels_add.append(
                PauliErrorAdder(qidx=q_idxes, pauli_error=ep, gate_names=[gate_op])
            )
            if noisy_message_level > 2:
                print(f"Add {c}.{gate_op} pauli error {ep}")
    noise_channels += noise_channels_add


def gen_noise_channels(q_names, c_names, noise_param, noisy_message_level=1):
    hash_val = DeepHash(noise_param)[noise_param]
    gate_dict = {"cz": "Z", "h": "H"}
    noise_param_mindquantum = {}
    for key, val in noise_param.items():
        if key in ["t1", "t2_se"]:
            noise_param_mindquantum[key] = val
        elif key in [
            "sq_len",
            "tq_len",
            "sq_pauli_error",
            "tq_pauli_error",
        ]:
            noise_param_mindquantum[key] = {}
            for gate, param in val.items():
                gate = gate_dict[gate] if gate in gate_dict else gate
                noise_param_mindquantum[key][gate] = param
        elif key in ["basis_gates"]:
            noise_param_mindquantum[key] = []
            for gate in noise_param[key]:
                gate = gate_dict[gate] if gate in gate_dict else gate
                noise_param_mindquantum[key].append(gate)
    t1 = noise_param_mindquantum["t1"]
    t2_se = noise_param_mindquantum["t2_se"]
    sq_len = noise_param_mindquantum["sq_len"]
    tq_len = noise_param_mindquantum["tq_len"]
    sq_pauli_error = noise_param_mindquantum["sq_pauli_error"]
    tq_pauli_error = noise_param_mindquantum["tq_pauli_error"]
    noise_channels = []
    add_sq_pauli_error_mindquantum(
        noise_channels, q_names, sq_pauli_error, noisy_message_level
    )
    add_tq_pauli_error_mindquantum(
        noise_channels, q_names, c_names, tq_pauli_error, noisy_message_level
    )
    add_thermal_noise_mindquantum(
        noise_channels, q_names, c_names, t1, t2_se, sq_len, tq_len, noisy_message_level
    )
    return noise_channels, hash_val


def run_mqvector_evolution_idx(circ, seed, shots, idxs):
    sim = mq_simu.Simulator("mqvector", circ.n_qubits, seed=seed)
    result = []
    for _ in range(shots):
        sim.reset()
        sim.apply_circuit(circ)
        result.append([sim.get_pure_state_vector()[idxs]])
    return np.array(result)


def run_mqvector_shots_idx(circ, seed, shots, pool_size, idxs, noisy=True):
    batchs = [len(i) for i in np.array_split(np.arange(shots), pool_size)]
    rng = np.random.default_rng(seed)
    seeds = rng.random(pool_size) * 2**20
    if noisy:
        print(f"n_qubits: {circ.n_qubits}, c_thread: {c_thread}, batchs: {batchs}")
    pool = multiprocessing.Pool(pool_size)
    tasks = []
    for idx, batch in enumerate(batchs):
        tasks.append(
            pool.apply_async(
                run_mqvector_evolution_idx, (circ, int(seeds[idx]), batch, idxs)
            )
        )
    pool.close()
    pool.join()
    results = np.concatenate([task.get() for task in tasks], axis=0)
    return results
