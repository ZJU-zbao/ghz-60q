import re

import numpy as np
import quaternion
from scipy.stats import unitary_group

I2 = np.diag([1, 1])
sigmaX = np.array([[0, 1], [1, 0]])
sigmaY = np.array([[0, -1j], [1j, 0]])
sigmaZ = np.array([[1, 0], [0, -1]])

q_names = {
    20: [
        "q1_1",
        "q1_3",
        "q1_5",
        "q1_7",
        "q1_9",
        "q3_9",
        "q3_7",
        "q3_5",
        "q3_3",
        "q5_3",
        "q5_5",
        "q5_7",
        "q5_9",
        "q7_9",
        "q7_7",
        "q7_5",
        "q7_3",
        "q7_1",
        "q5_1",
        "q3_1",
    ],
    18: [
        "q1_1",
        "q1_3",
        "q1_5",
        "q3_1",
        "q3_3",
        "q3_5",
        "q5_1",
        "q5_3",
        "q5_5",
        "q7_1",
        "q7_3",
        "q7_5",
        "q9_1",
        "q9_3",
        "q9_5",
        "q11_1",
        "q11_3",
        "q11_5",
    ],
    16: [
        "q1_1",
        "q1_3",
        "q1_5",
        "q1_7",
        "q3_7",
        "q3_5",
        "q3_3",
        "q5_3",
        "q5_5",
        "q5_7",
        "q7_7",
        "q7_5",
        "q7_3",
        "q7_1",
        "q5_1",
        "q3_1",
    ],
    14: [
        "q1_1",
        "q1_3",
        "q1_5",
        "q1_7",
        "q3_7",
        "q5_7",
        "q5_5",
        "q3_5",
        "q3_3",
        "q5_3",
        "q7_3",
        "q7_1",
        "q5_1",
        "q3_1",
    ],
    12: [
        "q1_1",
        "q1_3",
        "q1_5",
        "q1_7",
        "q3_7",
        "q5_7",
        "q5_5",
        "q3_5",
        "q3_3",
        "q5_3",
        "q5_1",
        "q3_1",
    ],
    10: [
        "q1_1",
        "q1_3",
        "q1_5",
        "q1_7",
        "q1_9",
        "q3_9",
        "q3_7",
        "q3_5",
        "q3_3",
        "q3_1",
    ],
    8: [
        "q1_1",
        "q1_3",
        "q1_5",
        "q1_7",
        "q3_7",
        "q3_5",
        "q3_3",
        "q3_1",
    ],
    6: ["q1_1", "q1_3", "q1_5", "q3_5", "q3_3", "q3_1"],
    4: ["q3_5", "q3_3", "q5_3", "q5_5"],
    2: ["q3_5", "q3_3"],
}

c_names_sensing = {
    20: [
        "c1_2",
        "c1_4",
        "c1_6",
        "c1_8",
        "c2_9",
        "c3_8",
        "c3_6",
        "c3_4",
        "c4_3",
        "c5_4",
        "c5_6",
        "c5_8",
        "c6_9",
        "c7_8",
        "c7_6",
        "c7_4",
        "c7_2",
        "c6_1",
        "c4_1",
        "c2_1",
    ],
    18: [
        "c1_2",
        "c1_4",
        "c2_5",
        "c4_5",
        "c6_5",
        "c8_5",
        "c10_5",
        "c11_4",
        "c11_2",
        "c10_1",
        "c9_2",
        "c8_3",
        "c7_2",
        "c6_1",
        "c5_2",
        "c4_3",
        "c3_2",
        "c2_1",
    ],
    16: [
        "c1_2",
        "c1_4",
        "c1_6",
        "c2_7",
        "c3_6",
        "c3_4",
        "c4_3",
        "c5_4",
        "c5_6",
        "c6_7",
        "c7_6",
        "c7_4",
        "c7_2",
        "c6_1",
        "c4_1",
        "c2_1",
    ],
    14: [
        "c1_2",
        "c1_4",
        "c1_6",
        "c2_7",
        "c4_7",
        "c5_6",
        "c4_5",
        "c3_4",
        "c4_3",
        "c6_3",
        "c7_2",
        "c6_1",
        "c4_1",
        "c2_1",
    ],
    12: [
        "c1_2",
        "c1_4",
        "c1_6",
        "c2_7",
        "c4_7",
        "c5_6",
        "c4_5",
        "c3_4",
        "c4_3",
        "c5_2",
        "c4_1",
        "c2_1",
    ],
    8: [
        "c1_2",
        "c1_4",
        "c1_6",
        "c2_7",
        "c3_6",
        "c3_4",
        "c3_2",
        "c2_1",
    ],
    4: ["c3_4", "c4_3", "c5_4", "c4_5"],
}

cat_pattern_sensing = {
    20: {
        0: ["q5_5"],
        1: ["c4_5"],
        2: ["c5_4", "c3_6"],
        3: ["c3_4", "c5_6", "c6_3", "c2_7"],
        4: ["c6_5", "c2_5", "c2_3", "c6_7", "c5_2", "c5_8"],
        5: ["c3_2", "c3_8", "c1_2", "c1_8", "c7_2", "c7_8"],
    },
    18: {
        0: ["q5_3"],
        1: ["c6_3"],
        2: ["c7_2", "c4_3"],
        3: ["c7_4", "c5_2", "c8_1", "c2_3"],
        4: ["c8_3", "c5_4", "c3_2", "c8_5", "c10_1", "c1_2"],
        5: ["c3_4", "c1_4", "c10_3", "c10_5"],
    },
    16: {
        0: ["q5_5"],
        1: ["c4_5"],
        2: ["c5_4", "c3_4"],
        3: ["c5_6", "c3_6", "c5_2", "c3_2"],
        4: ["c6_3", "c6_5", "c2_3", "c2_5", "c6_1", "c6_7"],
        5: ["c2_1", "c2_7"],
    },
    14: {
        0: ["q5_5"],
        1: ["c4_5"],
        2: ["c3_4", "c5_4"],
        3: ["c5_2", "c3_2", "c5_6", "c3_6"],
        4: ["c2_1", "c2_3", "c2_5", "c2_7", "c6_1", "c6_3"],
    },
    12: {
        0: ["q3_3"],
        1: ["c3_4"],
        2: ["c4_3", "c2_5"],
        3: ["c2_3", "c4_5", "c5_2", "c1_6"],
        4: ["c3_2", "c3_6", "c1_2", "c5_6"],
    },
    10: {
        0: ["q3_3"],
        1: ["c3_4"],
        2: ["c3_2", "c3_6"],
        3: ["c2_1", "c2_3", "c2_5", "c2_7"],
        4: ["c3_8", "c1_8"],
    },
    8: {
        0: ["q3_3"],
        1: ["c3_4"],
        2: ["c3_2", "c3_6"],
        3: ["c2_1", "c2_3", "c2_5", "c2_7"],
    },
    6: {
        0: ["q3_3"],
        1: ["c3_4"],
        2: ["c3_2", "c2_5"],
        3: ["c2_1", "c2_3"],
    },
    4: {
        0: ["q3_3"],
        1: ["c3_4"],
        2: ["c4_5", "c4_3"],
    },
    2: {0: ["q3_3"], 1: ["c3_4"]},
}


def gen_ghz_circuit(cat_pattern, add_last_h=True):
    """
    generate a ghz state with AFM type:
        $|01\dots\rangle + |10\dots\rangle$
    if add_last_h is True, return the whole circuit.
    Otherwise return more information
    """
    cat_circuit = {}
    qobj = cat_pattern[0][0]
    entangled_qs = [qobj]
    layer_circuit = {qobj: ["H"]}
    layer_i = 0
    last_qnames = []
    for layer_idx in np.arange(1, np.max(list(cat_pattern.keys())) + 1):
        for qname_tmp in last_qnames:
            layer_circuit.update({qname_tmp: ["H"]})
        last_qnames = []
        layer_cz_circuit = {}
        for qobj in cat_pattern[layer_idx]:
            if qobj.startswith("c"):
                layer_cz_circuit.update({qobj: ["cz"]})
                qname_tmps = c2q(qobj)
                for q_name in qname_tmps:
                    if q_name not in entangled_qs:
                        last_qnames.append(q_name)
        for qname_tmp in last_qnames:
            layer_circuit.update({qname_tmp: ["X", "H"]})
        cat_circuit.update({layer_i: layer_circuit, layer_i + 1: layer_cz_circuit})
        entangled_qs.extend(last_qnames)
        layer_circuit = {}
        layer_i += 2

    if add_last_h:
        for qname_tmp in last_qnames:
            layer_circuit.update({qname_tmp: ["H"]})
        cat_circuit.update({layer_i: layer_circuit})
        return cat_circuit
    else:
        return cat_circuit, last_qnames, layer_i


def gen_parity_circuit(
    q_names,
    cat_pattern,
):
    """
    generate ghz state and add tomography pulse to measure parity
    --------
    parity $\langle \mathcal{P}\rangle$ as function of $\gamma$
    """
    cat_circuit, last_qnames, layer_i = gen_ghz_circuit(
        cat_pattern, q_names, add_last_h=False
    )
    layer_circuit = {}
    for qi, qobj in enumerate(q_names):
        layer_circuit_qi = []
        if qobj in last_qnames:
            layer_circuit_qi += ["H"]
        if qobj in q_names[1::2]:
            layer_circuit_qi += ["X"]
        layer_circuit_qi += [("Rz", "gamma"), ("Rx", np.pi / 2)]
        layer_circuit.update({qobj: layer_circuit_qi})
    cat_circuit.update({layer_i: layer_circuit})
    return cat_circuit


def matrix_to_quaternion(m):
    i = -1j * sigmaX
    j = -1j * sigmaY
    k = -1j * sigmaZ
    return quaternion.from_float_array(
        np.abs(
            [
                np.trace(m) / 2,
                -np.trace(np.dot(m, i) / 2),
                -np.trace(np.dot(m, j) / 2),
                -np.trace(np.dot(m, k) / 2),
            ]
        )
    )


def quaternion_to_matrix(q):
    i = -1j * sigmaX
    j = -1j * sigmaY
    k = -1j * sigmaZ
    return q.w * I2 + q.x * i + q.y * j + q.z * k


def rotation(alpha, theta, phi):
    return np.quaternion(
        np.cos(alpha / 2),
        np.sin(alpha / 2) * np.sin(theta) * np.cos(phi),
        np.sin(alpha / 2) * np.sin(theta) * np.sin(phi),
        np.sin(alpha / 2) * np.cos(theta),
    )


def c2q(name):
    pattern = re.compile(r"c(\d+)_(\d+)")
    idx1, idx2 = pattern.findall(name)[0]
    idx1, idx2 = int(idx1), int(idx2)
    if idx1 % 2:
        return [f"q{idx1}_{idx2-1}", f"q{idx1}_{idx2+1}"]
    else:
        return [f"q{idx1-1}_{idx2}", f"q{idx1+1}_{idx2}"]


class RotationCompiler:
    def __init__(self):
        self._quaternion = None
        self.init_rotation()

    def init_rotation(self):
        self._quaternion = rotation(0, 0, 0)

    def qiskit_u(self, theta, phi, angle_lambda):
        _U = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * angle_lambda) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + angle_lambda)) * np.cos(theta / 2),
                ],
            ]
        )
        self._quaternion = matrix_to_quaternion(_U) * self._quaternion

    def random_unitary(self, size=1, random_state=None):
        if size == 1:
            self._quaternion = (
                matrix_to_quaternion(
                    unitary_group.rvs(2, size=size, random_state=random_state)
                )
                * self._quaternion
            )
        else:
            self._quaternion = (
                matrix_to_quaternion(
                    unitary_group.rvs(2, size=size, random_state=random_state)[-1]
                )
                * self._quaternion
            )

    def rotation(self, alpha, theta, phi):
        self._quaternion = rotation(alpha, theta, phi) * self._quaternion

    def Rx(self, alpha):
        self.rotation(alpha, np.pi / 2, 0)

    def Ry(self, alpha):
        self.rotation(alpha, np.pi / 2, np.pi / 2)

    def Rz(self, alpha):
        self.rotation(alpha, 0, 0)

    def plus_gate(self, gate):
        if gate == "X":
            self.rotation(np.pi, np.pi / 2, 0)
        elif gate == "-X":
            self.rotation(np.pi, np.pi / 2, np.pi)
        elif gate == "Y":
            self.rotation(np.pi, np.pi / 2, np.pi / 2)
        elif gate == "-Y":
            self.rotation(np.pi, np.pi / 2, np.pi * 3 / 2)
        elif gate == "X/2":
            self.rotation(np.pi / 2, np.pi / 2, 0)
        elif gate == "X/2+Y/2":
            self.rotation(np.pi / 2, np.pi / 2, np.pi / 4)
        elif gate == "Y/2":
            self.rotation(np.pi / 2, np.pi / 2, np.pi / 2)
        elif gate == "-X/2+Y/2":
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 3 / 4)
        elif gate == "-X/2":
            self.rotation(np.pi / 2, np.pi / 2, np.pi)
        elif gate == "-X/2-Y/2":
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 5 / 4)
        elif gate == "-Y/2":
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 3 / 2)
        elif gate == "X/2-Y/2":
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 7 / 4)
        elif gate == "H":
            self.rotation(np.pi, 0, 0)
            self.rotation(np.pi / 2, np.pi / 2, np.pi / 2)
        elif gate == "Z":
            self.rotation(np.pi, 0, 0)
        elif gate == "Z/2":
            self.rotation(np.pi / 2, 0, 0)
        elif gate == "-Z/2":
            self.rotation(-np.pi / 2, 0, 0)
        elif gate == "I":
            self.rotation(0, 0, 0)
        else:
            raise Exception(f"gate {gate} not supported!!")

    def add_gates(self, gates):
        for gate in gates:
            if isinstance(gate, str):
                self.plus_gate(gate)
            elif isinstance(gate, tuple):
                self.__getattribute__(gate[0])(gate[1])

    def compile_gate(self):
        return quaternion_to_matrix(self._quaternion)