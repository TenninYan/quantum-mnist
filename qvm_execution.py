
import gate_ansatz
from pyquil.api import QVMConnection, Job
from numpy import linspace

qvm = QVMConnection()

data = linspace(0,1,16)
depth = 1
N_qubit = gate_ansatz.N_qubit

init_theta = gate_ansatz.get_init_theta(depth)
p = gate_ansatz.input_gates(data) + gate_ansatz.output_gates(init_theta,depth)
classical_reg = [i for i in range(N_qubit)]
for i in range(N_qubit):
    p.measure(i,i)

print(p)

result = qvm.run(p,classical_reg,trials=1024)

print(result)
