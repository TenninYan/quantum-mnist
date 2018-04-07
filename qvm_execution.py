"""
runs qvm based on the gate ansatz.py
"""

import gate_ansatz
from pyquil.api import QVMConnection, Job
from pyquil.gates import *
import numpy as np
import scipy as sp

qvm = QVMConnection()

depth = 8
N_qubit = gate_ansatz.N_qubit
init_theta = gate_ansatz.get_init_theta(depth)

def get_output(data,theta):
    p = gate_ansatz.input_gates(data) + gate_ansatz.output_gates(init_theta,depth)
    classical_reg = [i for i in range(N_qubit)]
    result = qvm.expectation(p,[Z(i) for i in range(N_qubit)])  
    return (np.array(result)+1)/2
